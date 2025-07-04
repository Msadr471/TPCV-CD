import torch
from thop import profile, clever_format
from models.change_classifier import ChangeClassifier

# Dummy input
ref = torch.randn(1, 3, 256, 256)
test = torch.randn(1, 3, 256, 256)

# Model init
model = ChangeClassifier(
    bkbn_name="efficientnet_b4",
    weights=None,
    output_layer_bkbn="3",
    freeze_backbone=False
)

model.eval()

# Forward pass test
with torch.no_grad():
    out = model(ref, test)

print("✅ Forward pass OK")
print("Test shape:", test.shape)
print("Ref shape:", ref.shape)

print("Output shape:", out.shape)

# Params count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# FLOPs + MACs
flops, params = profile(model, inputs=(ref, test), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}")
print(f"THOP Params: {params}")
