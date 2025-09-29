#!/usr/bin/env python3
"""
RGB color generation tool for MCP experiment.
Usage: python rgb.py <red> <green> <blue>
"""

import sys
import argparse
import json


def generate_rgb_color(red: int, green: int, blue: int) -> dict:
    """Generate RGB color representation."""
    
    # Validate RGB values
    for value, name in [(red, 'red'), (green, 'green'), (blue, 'blue')]:
        if not 0 <= value <= 255:
            raise ValueError(f"{name} value must be between 0 and 255, got {value}")
    
    # Generate color information
    hex_color = f"#{red:02x}{green:02x}{blue:02x}"
    
    # Color name mapping for common colors
    color_names = {
        (255, 0, 0): "red",
        (0, 255, 0): "green", 
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 0, 255): "magenta",
        (0, 255, 255): "cyan",
        (255, 255, 255): "white",
        (0, 0, 0): "black",
        (255, 165, 0): "orange",
        (128, 0, 128): "purple"
    }
    
    color_name = color_names.get((red, green, blue), "custom")
    
    result = {
        "rgb": [red, green, blue],
        "hex": hex_color,
        "name": color_name,
        "success": True
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate RGB color representation')
    parser.add_argument('red', type=int, help='Red value (0-255)')
    parser.add_argument('green', type=int, help='Green value (0-255)')  
    parser.add_argument('blue', type=int, help='Blue value (0-255)')
    parser.add_argument('--format', choices=['json', 'text'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        result = generate_rgb_color(args.red, args.green, args.blue)
        
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"RGB({args.red}, {args.green}, {args.blue}) = {result['hex']} ({result['name']})")
            
    except ValueError as e:
        error_result = {
            "error": str(e),
            "success": False
        }
        
        if args.format == 'json':
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()