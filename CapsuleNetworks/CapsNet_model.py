import torch
import torch.nn.functional as F
from torch import nn

# im_size = 32
im_size = 128

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        # Premières couches de convulution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1) # voir quelle valeur de stride
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=9, stride=2) # / padding mettre etc.

        # Adjust kernel size and stride to match expected output
        self.primary_capsules = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2)

        self.num_primary_capsules = 32 * 6 * 6  # Nombre de capsules

        # Update the digit capsule layer to match the actual number of primary capsules (8192)
        self.digit_capsules = nn.ModuleList(
            [nn.Linear(147456, 2) for _ in range(10)]  # Each digit capsule takes 18432 as input now
        )
        

        # Reconstruction network
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, im_size * im_size),  # im_size = taille d'une dimension de l'image
            nn.Sigmoid()
        )

    def squash(self, tensor, dim=-1, epsilon=1e-8):
        # Fonction squash pour avoir des vecteurs de norme <= 1 (probas)
        #
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + epsilon) # on ajoute epsilon pour pas diviser par 0

    def forward(self, x):
        # Initial convolutions
        out = F.relu(self.conv1(x))
        print(f"Shape after conv1: {out.shape}")
        out = F.relu(self.conv2(out))
        
        # Primary capsules
        out = F.relu(self.primary_capsules(out))
        print(f"Shape after primary capsules: {out.shape}")  # Should be [batch_size, 256, 6, 6]
    
        # Reshape out to match the expected size for primary capsules
        out = out.view(out.size(0), -1, 8)  # Reshape to [batch_size, num_primary_capsules, capsule_dim]
        print(f"Shape after reshaping: {out.shape}")  # Should be [batch_size, 9216, 8]
    
        out = self.squash(out)
        print(f"Shape after squash: {out.shape}")  # Should be [batch_size, 9216, 8]
    
        # Flatten to pass to digit capsules
        out = out.view(out.size(0), -1) 
        print(f"Shape for digit capsules: {out.shape}")
    
        # Now, stack outputs from digit capsules
        u_hat = torch.stack([capsule(out) for capsule in self.digit_capsules], dim=1)
        print(f"Shape of u_hat after stacking: {u_hat.shape}")  
    
        # Routing
        v_j = self.routing(u_hat)
    
        # Reconstruction
        reconstruction_input = v_j.view(v_j.size(0), -1)
        reconstruction = self.reconstruction_layers(reconstruction_input)
    
        return v_j, reconstruction


    def routing(self, u_hat, num_iterations=3):
        
        b_ij = torch.zeros(u_hat.size(0), u_hat.size(1), device=u_hat.device)  
    
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1).unsqueeze(2)  
            s_j = (c_ij * u_hat).sum(dim=1)  
            v_j = self.squash(s_j)
            if iteration < num_iterations - 1:
                delta_b_ij = (u_hat * v_j.unsqueeze(1)).sum(dim=2) 
                b_ij = b_ij + delta_b_ij
        return v_j


# Initialiser le modèle
model = CapsuleNetwork()

# Summary of the model
from torchsummary import summary
summary(model, input_size=(1, 128, 128))

