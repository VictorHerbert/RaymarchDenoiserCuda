import bpy
import os
import sys
import re
import glob

# === Parse command-line args after '--' ===
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

if len(argv) != 1:
    print("Please pass exactly one sample value, e.g., '-- 64'")
    sys.exit(1)

try:
    samples = int(argv[0])
except ValueError:
    print("Sample value must be an integer.")
    sys.exit(1)

# Output base directory
cwd = os.getcwd()
output_root = cwd + "/render"
blend_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
output_path = bpy.path.abspath(os.path.join(output_root, blend_name, f"{samples}"))

# === Compositor Setup Function ===
def setup_compositor(output_path):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rlayers = tree.nodes.new(type='CompositorNodeRLayers')
    composite = tree.nodes.new(type='CompositorNodeComposite')
    file_output = tree.nodes.new(type='CompositorNodeOutputFile')

    file_output.base_path = output_path
    file_output.format.file_format = 'PNG'
    file_output.file_slots.clear()

    slots = {
        "render": "Image",
        "normal": "Normal",
        "albedo": "DiffCol",
        "depth": "Depth"
    }

    for slot_name in slots:
        file_output.file_slots.new(slot_name)

    view_layer = bpy.context.view_layer
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_z = True

    for slot_name, output_name in slots.items():
        tree.links.new(rlayers.outputs[output_name], file_output.inputs[slot_name])
    tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])

# === Render ===
print(f"\n🟢 Rendering with {samples} samples")
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = samples

setup_compositor(output_path)
bpy.ops.render.render(write_still=True)

# === Rename output files to remove frame numbers ===
pattern = re.compile(r"(\d{4})(?=\.png$)")
for file_path in glob.glob(os.path.join(output_path, "*.png")):
    new_path = pattern.sub("", file_path)
    if new_path != file_path:
        os.rename(file_path, new_path)

print(f"✅ Rendered: {samples} samples to {output_path}")