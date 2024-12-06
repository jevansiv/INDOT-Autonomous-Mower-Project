import torch
import torch.nn as nn
import torch.nn.functional as F

class Nav_Model(nn.Module):
    def __init__(self, in_scalars=1, num_outputs=1, embed_dim=128, num_heads=4, num_layers=4, seq_len=32, img_size=(1, 32, 32), hidden_dim=256, max_history=32):
        super(Nav_Model, self).__init__()

        self.max_history = max_history
        self.history_count=0
        self.past_imgs = None
        self.past_scalars = []

        # Convolutional layers to process each image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 16, 16)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 8, 8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 4, 4)
        )

        self.scalar_layers = nn.ModuleList()
        for i in range(in_scalars):
            self.scalar_layers.append(nn.Sequential(
                nn.Linear(in_features=1, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=256)
            ))

        # Calculate the flattened size after convolutional layers
        conv_output_size = 64 * 4 * 4 + 256*in_scalars

        # Embedding for the output of convolutional layers
        self.embedding = nn.Linear(conv_output_size, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to get a scalar value
        self.fc_out = nn.Linear(embed_dim, 1)

    def reset_state(self):
        self.history_count = 0
        self.past_imgs = None
        self.past_scalars = []


    def forward(self, image, scalars=[]): # should be shape (1, 32, 32) and [(1), (1), ...]

        # good luck figuring out the shape stuff
        if self.history_count == 0:

            for i in scalars:
                self.past_scalars.append(i.unsqueeze(0).unsqueeze(0))

            image = image.unsqueeze(0).unsqueeze(0) # make (1, 1, 1, 32, 32)
            self.past_imgs = image
        else:
            for i, sc in enumerate(self.past_scalars):
                # sc is shape (1, seq, 1). scalars[i] is shape (1)
                # print("sc", sc.shape)
                sc = sc[0, :, 0]

                self.past_scalars[i] = torch.cat([sc, scalars[i]])
                self.past_scalars[i] = self.past_scalars[i].unsqueeze(0).unsqueeze(2)
                # print("scc", self.past_scalars[i].shape)

            # print("pi", self.past_imgs.shape,image.shape)
            self.past_imgs = torch.cat([self.past_imgs, image.unsqueeze(0).unsqueeze(0)], dim=1)
            self.past_imgs.unsqueeze(0)

            if self.history_count >= self.max_history:
                # print("Cropping")
                self.past_imgs = self.past_imgs[:, 1::, :, :, :]
                for i in range(len(self.past_scalars)):
                    self.past_scalars[i] = self.past_scalars[i][:, 1::, :]
        self.history_count += 1

        # print("im", self.past_imgs.shape)
        # print("ps", self.past_scalars[0].shape)

        # x is of shape (batch_size, seq_len, 1, 32, 32)
        batch_size, seq_len, _, height, width = self.past_imgs.shape

        # Process each image through convolutional layers
        image = self.past_imgs.view(batch_size * seq_len, 1, height, width)  # Shape: (batch_size * seq_len, 1, 32, 32)
        image = self.conv_layers(image)  # Shape: (batch_size * seq_len, 64, 4, 4)
        
        # Flatten the output of convolutional layers
        x = image.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, 64 * 4 * 4)


        if len(scalars) > 0:
            # print("x", x.shape)
            ss = [x]
            for l, s in zip(self.scalar_layers, self.past_scalars):
                # print("ss", s.shape)
                # s = s.unsqueeze(0)
                m = l(s)
                # print("mm", m.shape)
                ss.append(m)
            x = torch.cat(ss, dim=2)
            # print("xx", x.shape)


        # Embed the flattened features
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Transformer expects input in (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, embed_dim)

        # Pass through the transformer encoder
        x = self.transformer(x)  # Shape: (seq_len, batch_size, embed_dim)

        # Take the mean across the sequence dimension
        x = x.mean(dim=0)  # Shape: (batch_size, embed_dim)

        # Final output is a scalar
        out = self.fc_out(x)  # Shape: (batch_size, 1)
        return out


if __name__ == "__main__":
    import numpy as np

    model = Nav_Model(in_scalars=1, num_outputs=1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("num parameters", str(int(params/10**3)) + "K")

    for i in range(33):
        sample_input_img = torch.randn(1, 32, 32) #
        sample_input_scalar = torch.randn(1)
        output = model(sample_input_img, [sample_input_scalar])
        print("out->", output)


