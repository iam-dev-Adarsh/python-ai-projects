from PIL import Image, ImageDraw

WIDTH = 1600
HEIGHT = 880

img = Image.new('RGB', (WIDTH, HEIGHT), (255, 255, 255))
draw = ImageDraw.Draw(img)

# Smooth, continuous racing circuit
track_path = [
    (250, 150), (1350, 150), 
    (1450, 220), (1450, 660), 
    (1350, 730), (1050, 730), 
    (920, 520), (700, 520),
    (580, 730), (250, 730), 
    (150, 660), (150, 220), (250, 150)
]

# Outer dark border, red curb, and wide 120px grey asphalt road
draw.line(track_path, fill=(30, 30, 30), width=160, joint='curve')
draw.line(track_path, fill=(220, 38, 38), width=145, joint='curve')
draw.line(track_path, fill=(100, 105, 115), width=120, joint='curve')

# Checkered start/finish line at X=800, Y=150
for y in range(90, 210, 15):
    color = (255, 255, 255) if (y // 15) % 2 == 0 else (0, 0, 0)
    draw.rectangle([795, y, 805, y + 15], fill=color)

img.save('map.png')
print("Successfully generated new map.png!")
