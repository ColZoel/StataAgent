from pathlib import Path

# Use whatever path you've been entering — type it as you would normally
test_path = Path(r"C:\Program Files\StataNow19\utilities")  # adjust to yours

print(f"Path object:        {test_path}")
print(f"Path exists:        {test_path.exists()}")
print(f"Is directory:       {test_path.is_dir()}")
print(f"pystata subdir:     {test_path / 'pystata'}")
print(f"pystata exists:     {(test_path / 'pystata').exists()}")
print(f"pystata is dir:     {(test_path / 'pystata').is_dir()}")

# # Also list what's actually there:
# if test_path.exists():
#     print(f"\nContents of {test_path}:")
#     for item in test_path.iterdir():
#         print(f"  {item.name}")