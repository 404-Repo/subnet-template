import torch
import time

from validator import _render_obj, _run_validation

start = time.time()
images = _render_obj(file_name="output.ply")
inter = time.time()
score = _run_validation(
    prompt="A Golden Poison Dart Frog", images=images, device=torch.device("cpu")
)
end = time.time()

print(score)
print(inter - start)
print(end - inter)

# import matplotlib.pyplot as plt
#
# for x in range(4):
#     plt.imshow(r[x])
#     plt.savefig(f'image{x}.png')
