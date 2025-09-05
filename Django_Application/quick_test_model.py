import os, sys, torch
sys.path.append(os.getcwd())

# Fix for Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_settings.settings")

from django.conf import settings
import django
django.setup()

from ml_app.views import load_global_model, validation_dataset, train_transforms, device

def run_test():
    print("Device:", device)
    vids_dir = os.path.join(settings.PROJECT_DIR, 'uploaded_videos')
    if not os.path.exists(vids_dir):
        print("No uploaded_videos directory found")
        return
    vids = [os.path.join(vids_dir, f) for f in os.listdir(vids_dir) if f.endswith('.mp4')]
    if not vids:
        print("No test video found in uploaded_videos; please copy one there and run again.")
        return
    video = vids[0]
    print("Testing with:", video)
    ds = validation_dataset([video], sequence_length=60, transform=train_transforms)
    batch = ds[0]  # shape expected (1, seq, c,h,w)
    print("Dataset returned shape:", batch.shape)
    model = load_global_model()
    model.to(device)
    model.eval()
    with torch.inference_mode():
        fmap, logits = model(batch.to(device))
    print("fmap shape:", fmap.shape)
    print("logits shape:", logits.shape)
    print("logits (cpu):", logits.detach().cpu().numpy())
    sm = torch.nn.Softmax(dim=1)
    sm_out = sm(logits)
    print("softmax:", sm_out.detach().cpu().numpy())
    print("softmax stats: min, max, mean:", float(sm_out.min()), float(sm_out.max()), float(sm_out.mean()))

if __name__ == '__main__':
    run_test()