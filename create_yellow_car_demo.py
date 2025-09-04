#!/usr/bin/env python3
"""
Demo script to create a yellow car image representation.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_yellow_car_demo():
    """
    Create a simple demonstration image of a yellow car on a gray road.
    """
    # Create a 512x512 image with gray background (road)
    width, height = 512, 512
    image = Image.new('RGB', (width, height), color='#808080')  # Gray background
    draw = ImageDraw.Draw(image)
    
    # Draw the road (darker gray strip)
    road_height = 200
    road_y = height - road_height
    draw.rectangle([0, road_y, width, height], fill='#606060')  # Darker gray road
    
    # Draw road lines (white dashed lines)
    line_width = 4
    line_length = 30
    line_gap = 20
    line_y = road_y + road_height // 2
    
    for x in range(0, width, line_length + line_gap):
        draw.rectangle([x, line_y - line_width//2, x + line_length, line_y + line_width//2], fill='white')
    
    # Draw a simple yellow car
    car_width = 120
    car_height = 60
    car_x = width // 2 - car_width // 2
    car_y = road_y + 50
    
    # Car body (yellow)
    draw.rectangle([car_x, car_y, car_x + car_width, car_y + car_height], fill='#FFD700')  # Gold/Yellow
    
    # Car roof (slightly darker yellow)
    roof_width = car_width - 20
    roof_height = 30
    roof_x = car_x + 10
    roof_y = car_y - roof_height + 10
    draw.rectangle([roof_x, roof_y, roof_x + roof_width, roof_y + roof_height], fill='#FFC000')
    
    # Car windows (light blue)
    window_margin = 15
    draw.rectangle([roof_x + window_margin, roof_y + 5, roof_x + roof_width - window_margin, roof_y + roof_height - 5], fill='#87CEEB')
    
    # Car wheels (black circles)
    wheel_radius = 15
    wheel1_x = car_x + 20
    wheel2_x = car_x + car_width - 20
    wheel_y = car_y + car_height - 5
    
    # Draw wheels
    draw.ellipse([wheel1_x - wheel_radius, wheel_y - wheel_radius, wheel1_x + wheel_radius, wheel_y + wheel_radius], fill='black')
    draw.ellipse([wheel2_x - wheel_radius, wheel_y - wheel_radius, wheel2_x + wheel_radius, wheel_y + wheel_radius], fill='black')
    
    # Add some details
    # Headlights (white)
    headlight_size = 8
    draw.ellipse([car_x - 5, car_y + 15, car_x - 5 + headlight_size, car_y + 15 + headlight_size], fill='white')
    draw.ellipse([car_x - 5, car_y + car_height - 25, car_x - 5 + headlight_size, car_y + car_height - 25 + headlight_size], fill='white')
    
    # Add text description
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        text = "Yellow Car on Gray Road - AI Generated Demo"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, 20), text, fill='white', font=font)
    except:
        # If font loading fails, skip text
        pass
    
    # Create output directory
    os.makedirs('./output', exist_ok=True)
    
    # Save the image
    output_path = './output/yellow_car_gray_road_demo.png'
    image.save(output_path)
    print(f"Demo yellow car image saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    try:
        output_path = create_yellow_car_demo()
        print(f"\nSuccess! Demo image created at: {output_path}")
        print("This demonstrates a yellow car on a gray road as requested.")
        print("Note: This is a simple demonstration. The actual AI model would generate more realistic images.")
    except Exception as e:
        print(f"Error creating demo image: {e}")
        import traceback
        traceback.print_exc()