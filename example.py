import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CausalEncoderBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=(1, 1),
    ):
        super().__init__()
        self.act = nn.ELU()

        time_stride, freq_stride = stride

        # As defined in the new diagram
        # First convolution: k=3, s=1, out_channels=in_channels
        self.conv1 = weight_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size=3))

        # Second convolution: dynamic kernel, applies stride, changes channels
        kernel_size_f = max(3, 2 * freq_stride)
        self.conv2 = weight_norm(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(3, kernel_size_f),
                      stride=stride))

        # Shortcut connection
        self.has_shortcut = (in_channels != out_channels) or (
            time_stride != 1) or (freq_stride != 1)
        if self.has_shortcut:
            # Projection layer to match channel dimensions
            self.projection = weight_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
            # If strided, use average pooling for downsampling on the shortcut path
            if time_stride > 1 or freq_stride > 1:
                self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
            else:
                self.avg_pool = nn.Identity()

    def forward(self, x):
        # x shape: (B, C, Time, Freq)

        # Shortcut connection path
        shortcut = x
        if self.has_shortcut:
            shortcut = self.avg_pool(shortcut)
            shortcut = self.projection(shortcut)

        # Main path
        # Causal padding for the first convolution (k=3, s=1)
        # Symmetrical padding for Freq (dim 3), causal (left) padding for Time (dim 2)
        x = F.pad(x, (1, 1, 2, 0))
        x = self.act(self.conv1(x))

        # Causal padding for the second convolution
        k_t = self.conv2.kernel_size[0]
        k_f = self.conv2.kernel_size[1]
        s_t = self.conv2.stride[0]
        # Calculate padding needed to maintain shape and causality
        pad_f = (k_f - 1) // 2
        pad_t = k_t - s_t  # Causal padding for a strided convolution
        x = F.pad(x, (pad_f, pad_f, pad_t, 0))
        x = self.act(x)
        x = self.conv2(x)

        return x + shortcut


class BottleneckBlock1D(nn.Module):
    """ The 1D Bottleneck block with a residual connection. """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = nn.ELU()

        # As per diagram, the intermediate channel is the max of in/out channels
        intermediate_channels = max(in_channels, out_channels)

        # 1x1 convolutions for feature transformation
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, intermediate_channels, kernel_size=1))
        self.conv2 = weight_norm(
            nn.Conv1d(intermediate_channels, out_channels, kernel_size=1))

        # Shortcut connection with projection if channels differ
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + shortcut


class SpectroStreamEncoder(nn.Module):
    """
    The full SpectroStream Encoder, built with the corrected helper blocks.
    It processes left and right channels in parallel before concatenation.
    """

    def __init__(self, C0=32, D=256):
        super().__init__()
        self.C0 = C0
        self.D = D

        # Initial Conv2D layer
        self.input_conv = weight_norm(nn.Conv2d(2, C0, kernel_size=(7, 7)))

        # The stream of six encoder blocks applied to each audio channel
        self.encoder_stream = nn.Sequential(
            CausalEncoderBlock(C0, 2 * C0, stride=(1, 2)),
            CausalEncoderBlock(
                2 * C0, 2 * C0, stride=(1, 2)
            ),  # This was 2C->4C in old diagram, but this is the corrected one
            CausalEncoderBlock(2 * C0, 4 * C0, stride=(1, 3)),
            CausalEncoderBlock(4 * C0, 4 * C0, stride=(1, 2)),
            CausalEncoderBlock(4 * C0, 4 * C0, stride=(1, 2)),
            CausalEncoderBlock(4 * C0, 8 * C0, stride=(2, 2)),
            CausalEncoderBlock(8 * C0, 8 * C0, stride=(2, 1)))

        # The block applied after concatenating the two stereo channels
        self.post_concat_block = CausalEncoderBlock(16 * C0,
                                                    8 * C0,
                                                    stride=(1, 1))

        # The final 1D bottleneck block
        # The input features are 40*C0 because the Freq dimension (5) is flattened into the channels (8*C0).
        self.bottleneck_block = BottleneckBlock1D(40 * C0, D)

    def _run_stream(self, x_ch):
        """ Helper function to run a single channel through the initial layers. """
        # Manual causal padding for the first conv (k=7)
        # Symmetrical for freq (3,3), causal for time (6,0)
        x_ch = F.pad(x_ch, (3, 3, 6, 0))
        z = self.input_conv(x_ch)
        z = self.encoder_stream(z)
        return z

    def forward(self, x_left_ch, x_right_ch):
        """
        Args:
            x_left_ch (Tensor): The STFT spectrogram for the left channel. 
                                Shape: (B, 2, T, F), where 2 is for real and imag parts.
            x_right_ch (Tensor): The STFT spectrogram for the right channel.
                                 Shape: (B, 2, T, F).
        Returns:
            Tensor: The final embedding. Shape: (B, D, T_out).
        """
        # Process each channel's spectrogram independently through the initial stream
        z_left = self._run_stream(x_left_ch)
        z_right = self._run_stream(x_right_ch)

        # "Channel-concatenate" step from the diagram
        # The two (B, 8*C0, T_out, F_out) tensors are concatenated on the channel dimension.
        z = torch.cat([z_left, z_right],
                      dim=1)  # Shape: (B, 16*C0, T_out, F_out)

        # Pass through the post-concatenation block
        z = self.post_concat_block(z)  # Shape: (B, 8*C0, T_out, F_out)

        # "Reshape" step: Flatten the Freq and Channel dimensions
        B, C, T_out, F_out = z.shape
        # (B, C, T_out, F_out) -> (B, T_out, C*F_out)
        z = z.permute(0, 2, 1, 3).reshape(B, T_out, C * F_out)
        # (B, T_out, C*F_out) -> (B, C*F_out, T_out) to match Conv1D input
        z = z.permute(0, 2, 1)  # Shape: (B, 40*C0, T_out)

        # Final "Bottleneck block" to produce the embedding
        embedding = self.bottleneck_block(z)  # Shape: (B, D, T_out)

        return embedding


class CausalDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.act = nn.ELU()

        time_stride, freq_stride = stride

        # First convolution: k=3, s=1, changes channels to N_out
        self.conv1 = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3))

        # Second convolution: TransposedConv2D for upsampling
        kernel_size_t = max(3, 2 * time_stride)
        kernel_size_f = max(3, 2 * freq_stride)
        self.transposed_conv = weight_norm(
            nn.ConvTranspose2d(out_channels,
                               out_channels,
                               kernel_size=(kernel_size_t, kernel_size_f),
                               stride=stride))

        # Shortcut connection
        self.has_shortcut = (in_channels != out_channels) or (
            time_stride != 1) or (freq_stride != 1)
        if self.has_shortcut:
            # Projection layer to match channel dimensions
            self.projection = weight_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
            # If strided, use upsampling on the shortcut path
            if time_stride > 1 or freq_stride > 1:
                self.upsample = nn.Upsample(scale_factor=stride,
                                            mode='nearest')
            else:
                self.upsample = nn.Identity()

    def forward(self, x):
        # Shortcut connection path
        shortcut = x
        if self.has_shortcut:
            shortcut = self.upsample(shortcut)
            shortcut = self.projection(shortcut)

        # Main path
        # We assume the decoder should also be causal for streaming generation
        # Causal padding for the first convolution
        x = F.pad(
            x,
            (1, 1, 2,
             0))  # Symmetrical padding for Freq, left padding for Time (k=3)
        x = self.act(self.conv1(x))
        x = self.act(x)

        # The transposed conv needs careful padding to align outputs, but the operation
        # itself is generally causal by nature (output at t depends on input at t).
        # We trim the output to maintain the correct length after upsampling.
        x = self.transposed_conv(x)

        # The output size of ConvTranspose2d can be slightly larger than desired.
        # We may need to crop it to match the shortcut's shape if they differ.
        if x.shape[2] != shortcut.shape[2]:
            x = x[:, :, :shortcut.shape[2], :]
        if x.shape[3] != shortcut.shape[3]:
            x = x[:, :, :, :shortcut.shape[3]]

        return x + shortcut


class SpectroStreamDecoder(nn.Module):
    """
    The full SpectroStream Decoder, the inverse of the Encoder.
    It takes an embedding and reconstructs the stereo spectrograms.
    """

    def __init__(self, C0=32, D=128):
        super().__init__()
        self.C0 = C0
        self.D = D

        # Initial 1D bottleneck block, inverting the encoder's bottleneck
        self.bottleneck_block = BottleneckBlock1D(D, 40 * C0)

        # The block that expands channels from 8*C0 to 16*C0 before the split
        self.pre_split_block = CausalDecoderBlock(8 * C0,
                                                  16 * C0,
                                                  stride=(1, 1))

        # The stream of six decoder blocks, shared by both stereo channels
        self.decoder_stream = nn.Sequential(
            CausalDecoderBlock(8 * C0, 8 * C0, stride=(2, 1)),
            CausalDecoderBlock(8 * C0, 4 * C0, stride=(2, 2)),
            CausalDecoderBlock(4 * C0, 4 * C0, stride=(1, 2)),
            CausalDecoderBlock(4 * C0, 4 * C0, stride=(1, 2)),
            CausalDecoderBlock(4 * C0, 2 * C0, stride=(1, 3)),
            CausalDecoderBlock(2 * C0, 2 * C0, stride=(1, 2)),
            CausalDecoderBlock(2 * C0, C0, stride=(1, 2)),
        )

        # Final output Conv2D layer, projecting back to 2 channels (real, imag)
        self.output_conv = weight_norm(nn.Conv2d(C0, 2, kernel_size=(7, 7)))

    def forward(self, embedding):
        """
        Args:
            embedding (Tensor): The input embedding from the encoder (or quantizer).
                                Shape: (B, D, T_out).
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the reconstructed left and
                                   right spectrograms. Shape of each is (B, 2, T_in, F_in).
        """
        # Pass through the initial bottleneck
        z = self.bottleneck_block(embedding)  # Shape: (B, 40*C0, T_out)

        # "Reshape" step: Un-flatten the channel and frequency dimensions
        B, _, T_out = z.shape
        # (B, 40*C0, T_out) -> (B, T_out, 8*C0, 5)
        z = z.permute(0, 2, 1).reshape(B, T_out, 8 * self.C0, 5)
        # (B, T_out, 8*C0, 5) -> (B, 8*C0, T_out, 5) to match Conv2D input
        z = z.permute(0, 2, 1, 3)

        # Pass through the pre-split block to expand channels to 16*C0
        z = self.pre_split_block(z)  # Shape: (B, 16*C0, T_out, 5)

        # "Channel-split" step: Split into two parallel streams for left and right channels
        z_left, z_right = torch.chunk(
            z, 2, dim=1)  # Each has shape (B, 8*C0, T_out, 5)

        # Pass each stream through the shared decoder blocks
        out_left = self.decoder_stream(z_left)
        out_right = self.decoder_stream(z_right)

        # Pass each stream through the final shared output convolution
        # Causal padding for the final conv (k=7)
        # Symmetrical for freq (3,3), causal for time (6,0)
        out_left = F.pad(out_left, (3, 3, 6, 0))
        out_right = F.pad(out_right, (3, 3, 6, 0))

        out_left_ch = self.output_conv(out_left)
        out_right_ch = self.output_conv(out_right)

        return out_left_ch, out_right_ch


class SpectroStream(nn.Module):
    """
    The main SpectroStream model that combines the Encoder and Decoder.
    It handles the end-to-end process from audio to audio.
    """

    def __init__(self, C0=32, D=128):
        super().__init__()
        self.encoder = SpectroStreamEncoder(C0, D)
        self.decoder = SpectroStreamDecoder(C0, D)

        # STFT parameters based on the paper's description
        # hop_length is 480 (from 100Hz frame rate at 48kHz).
        # win_length is 2x hop_length due to 2x overlapping factor.
        self.hop_length = 480
        self.win_length = 960
        self.n_fft = 960

    def forward(self, x):
        """
        The main forward pass of the SpectroStream model.

        Args:
            x (Tensor): The input stereo audio waveform.
                        Shape: (B, 2, T_samples), where B is batch size,
                               2 is for stereo, T_samples is the number of audio samples.
        Returns:
            Tensor: The reconstructed stereo audio waveform.
                    Its length will be slightly shorter than the input due to latency.
        """
        # --- STEP 1: Audio -> Spectrogram (STFT) ---
        # Process left and right channels separately
        x_left_stft = torch.stft(x[:, 0],
                                 n_fft=self.n_fft,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 return_complex=True)
        x_right_stft = torch.stft(x[:, 1],
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  return_complex=True)

        # --- STEP 2: Pre-process Spectrogram for the Network ---
        # "omit the highest (Nyquist) frequency to avoid odd number of bins"
        x_left_stft = x_left_stft[:, :
                                  -1, :]  # Shape becomes (B, 480, T_frames)
        x_right_stft = x_right_stft[:, :-1, :]

        # Convert complex spectrograms to 2-channel real tensors for the Conv2D layers
        # The shape needs to be (B, C, T, F) -> (B, 2, T_frames, Freq_bins)
        def to_channels(s):
            # stack real and imaginary parts as channels
            s_ch = torch.stack([s.real, s.imag], dim=1)
            # permute from (B, 2, F, T) to (B, 2, T, F)
            return s_ch.permute(0, 1, 3, 2)

        x_left_ch = to_channels(x_left_stft)
        x_right_ch = to_channels(x_right_stft)

        # --- STEP 3: Encoding ---
        # Pass the prepared spectrograms through the encoder
        embedding = self.encoder(x_left_ch, x_right_ch)

        # --- STEP 4: Quantization (Placeholder) ---
        # In a real codec, a quantizer (like RVQ) would be used here.
        # This step compresses the continuous embedding into discrete codes.
        # embedding -> codes -> quantized_embedding
        # For this autoencoder implementation, we pass it through directly.
        quantized_embedding = embedding

        # --- STEP 5: Decoder Look-ahead ---
        # "we give the decoder a one-embedding look-ahead by shifting its input"
        # We implement this by removing the first embedding from the sequence.
        decoder_input = quantized_embedding[:, :, 1:]

        # --- STEP 6: Decoding ---
        # The decoder takes the shifted embeddings and reconstructs the spectrograms
        out_left_stft_ch, out_right_stft_ch = self.decoder(decoder_input)

        # --- STEP 7: Post-process Spectrogram for Audio ---
        # Convert the 2-channel (real, imag) tensors back to complex numbers
        def to_complex(s_ch):
            # permute from (B, 2, T, F) back to (B, T, F, 2)
            s = s_ch.permute(0, 2, 3, 1)
            return torch.complex(s[..., 0], s[..., 1]).permute(0, 2, 1)

        out_left_stft_complex = to_complex(out_left_stft_ch)
        out_right_stft_complex = to_complex(out_right_stft_ch)

        # Add back the Nyquist frequency bin (as a row of zeros) before iSTFT
        out_left_stft = F.pad(out_left_stft_complex, (0, 0, 0, 1))
        out_right_stft = F.pad(out_right_stft_complex, (0, 0, 0, 1))

        # --- STEP 8: Spectrogram -> Audio (iSTFT) ---
        # Perform inverse STFT to get the final audio waveform
        out_left_wav = torch.istft(out_left_stft,
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length)
        out_right_wav = torch.istft(out_right_stft,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length)

        # --- STEP 9: Final Output ---
        # Combine the left and right channels into a final stereo waveform
        out_wav = torch.stack([out_left_wav, out_right_wav], dim=1)

        return out_wav


if __name__ == '__main__':
    model = SpectroStream()
    # for name, value in model.state_dict().items():
    #     print(name, value.shape)

    wav = torch.rand(1, 2, 48000)
    out_wav = model(wav)
    print(out_wav.shape)
