import os
from PIL import Image, ImageDraw, ImageFont

def draw_tree_flowchart(output_path="file_structure_flowchart.png"):
    # canvas dimensions
    width = 2400
    height = 2800
    bg_color = (255, 255, 255)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        font_root = ImageFont.truetype("arialbd.ttf", 36)
        font_folder = ImageFont.truetype("arialbd.ttf", 30)
        font_file = ImageFont.truetype("arial.ttf", 26)
        font_caption = ImageFont.truetype("ariali.ttf", 32)
    except:
        font_root = font_folder = font_file = font_caption = ImageFont.load_default()

    # Colors
    color_line = (0, 0, 0)
    color_dot = (0, 0, 0)
    color_text = (30, 30, 30)

    # Folder styling (matching sample picture amber/yellow)
    color_folder_fill = (245, 205, 80)
    color_folder_tab = (230, 185, 60)
    color_folder_stroke = (170, 130, 20)

    # Helper: Draw Folder Icon
    def draw_folder_icon(x, y, scale=1.0):
        w, h = int(48 * scale), int(36 * scale)
        tab_w, tab_h = int(22 * scale), int(10 * scale)
        # Tab
        draw.rectangle([x, y, x + tab_w, y + tab_h], fill=color_folder_tab, outline=color_folder_stroke, width=2)
        # Body
        draw.rectangle([x, y + tab_h - 2, x + w, y + h], fill=color_folder_fill, outline=color_folder_stroke, width=2)
        return w

    # Helper: Draw File Icon
    def draw_file_icon(x, y, icon_type="file", scale=1.0):
        w, h = int(34 * scale), int(42 * scale)
        fold = int(10 * scale)

        # Color mapping based on file type in sample
        if icon_type == "py":
            bg = (52, 120, 175)       # Python Blue
            border = (25, 70, 110)
        elif icon_type == "task" or icon_type == "model":
            bg = (130, 80, 180)      # Model Purple
            border = (80, 40, 120)
        elif icon_type == "img":
            bg = (215, 105, 55)      # Image Orange
            border = (150, 60, 25)
        elif icon_type == "txt" or icon_type == "csv":
            bg = (60, 155, 115)      # Document Green/Teal
            border = (30, 100, 70)
        elif icon_type == "tex":
            bg = (175, 65, 85)       # TeX Red/Crimson
            border = (110, 35, 50)
        else:
            bg = (120, 140, 160)
            border = (70, 85, 100)

        # Folded paper polygon
        points = [(x, y), (x + w - fold, y), (x + w, y + fold), (x + w, y + h), (x, y + h)]
        draw.polygon(points, fill=bg, outline=border, width=2)
        draw.polygon([(x + w - fold, y), (x + w - fold, y + fold), (x + w, y + fold)], fill=(230, 240, 250), outline=border, width=2)
        return w

    # Flattened layout tree structure with explicit relative positioning
    # We measure text widths to position arrows perfectly without overlap!

    # Helper to measure text width
    def get_text_width(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    # Data layout tree
    nodes_data = [
        {
            "name": "Dependencies",
            "type": "folder",
            "children": [
                {
                    "name": "Models",
                    "type": "folder",
                    "children": [
                        {"name": "face_landmarker.task", "type": "file", "icon": "task"}
                    ]
                },
                {"name": "requirements.txt", "type": "file", "icon": "txt"}
            ]
        },
        {
            "name": "Core_Modules",
            "type": "folder",
            "children": [
                {"name": "tracker.py", "type": "file", "icon": "py"},
                {"name": "virtual_keyboard.py", "type": "file", "icon": "py"},
                {"name": "text_pad.py", "type": "file", "icon": "py"},
                {"name": "quiz_module.py", "type": "file", "icon": "py"},
                {"name": "heatmap_generator.py", "type": "file", "icon": "py"},
                {"name": "gaze_cursor.py", "type": "file", "icon": "py"},
                {"name": "keyboardaccuracy.py", "type": "file", "icon": "py"},
                {"name": "latex.py", "type": "file", "icon": "py"}
            ]
        },
        {
            "name": "Outputs",
            "type": "folder",
            "children": [
                {
                    "name": "CSV_Logs",
                    "type": "folder",
                    "children": [
                        {"name": "experiment_covert_*.csv", "type": "file", "icon": "csv"},
                        {"name": "experiment_overt_*.csv", "type": "file", "icon": "csv"}
                    ]
                },
                {
                    "name": "Heatmaps",
                    "type": "folder",
                    "children": [
                        {"name": "experiment_*_heatmap.png", "type": "file", "icon": "img"}
                    ]
                },
                {
                    "name": "LaTeX_Reports",
                    "type": "folder",
                    "children": [
                        {"name": "experiment_*.tex", "type": "file", "icon": "tex"}
                    ]
                },
                {
                    "name": "Session_Reports",
                    "type": "folder",
                    "children": [
                        {"name": "gaze_log.csv", "type": "file", "icon": "csv"},
                        {"name": "quiz_results.csv", "type": "file", "icon": "csv"},
                        {"name": "session_report.png", "type": "file", "icon": "img"},
                        {"name": "session_quiz_report.png", "type": "file", "icon": "img"}
                    ]
                }
            ]
        },
        {
            "name": "Config_&_Metadata",
            "type": "folder",
            "children": [
                {"name": "calibration_profiles.json", "type": "file", "icon": "txt"},
                {"name": "calibration_profiles.csv", "type": "file", "icon": "csv"},
                {"name": "usage_stats.json", "type": "file", "icon": "txt"},
                {"name": "readme.md", "type": "file", "icon": "txt"}
            ]
        },
        {"name": "launcher.py", "type": "file", "icon": "py"},
        {"name": "main.py", "type": "file", "icon": "py"},
        {"name": "tracker_run.py", "type": "file", "icon": "py"}
    ]

    # Render Root
    root_x, root_y = 150, 120
    draw_folder_icon(root_x, root_y, scale=1.2)
    draw.text((root_x + 70, root_y - 2), "Eye_Tracking_System", fill=color_text, font=font_root)

    main_trunk_x = root_x + 35
    row_height = 65
    line_w = 4
    dot_r = 8

    # We will do a 2-pass approach:
    # Pass 1: compute y-coordinate for every leaf/node to ensure perfect spacing.
    # Pass 2: draw lines, icons, text, arrows.

    current_y = root_y + 110

    def compute_height(item):
        if item["type"] == "file" or "children" not in item:
            return 1
        total = 0
        for c in item["children"]:
            total += compute_height(c)
        return total

    # Draw function
    main_branch_y_list = []

    y_cursor = current_y

    for item in nodes_data:
        h_units = compute_height(item)
        item_y_center = y_cursor + ((h_units - 1) * row_height) / 2.0
        main_branch_y_list.append((item, item_y_center, y_cursor))
        y_cursor += h_units * row_height + 25  # padding between main sections

    # Draw Main Trunk Line
    first_y = main_branch_y_list[0][1]
    last_y = main_branch_y_list[-1][1]
    draw.line([(main_trunk_x, first_y), (main_trunk_x, last_y)], fill=color_line, width=line_w)

    def draw_recursive(item, start_x, center_y, top_y):
        # Draw dot on trunk
        draw.ellipse([start_x - dot_r, center_y - dot_r, start_x + dot_r, center_y + dot_r], fill=color_dot)
        
        # Horizontal line from trunk to item icon
        icon_x = start_x + 60
        draw.line([(start_x, center_y), (icon_x, center_y)], fill=color_line, width=line_w)

        if item["type"] == "folder":
            icon_w = draw_folder_icon(icon_x, center_y - 18, scale=1.05)
            text_x = icon_x + icon_w + 15
            draw.text((text_x, center_y - 16), item["name"], fill=color_text, font=font_folder)
            
            # If folder has children, draw arrow -> to sub-trunk
            if "children" in item and item["children"]:
                txt_w = get_text_width(item["name"], font_folder)
                arrow_start_x = text_x + txt_w + 25
                sub_trunk_x = arrow_start_x + 100
                
                # Arrow line
                draw.line([(arrow_start_x, center_y), (sub_trunk_x, center_y)], fill=color_line, width=line_w)
                # Arrow head
                draw.polygon([(sub_trunk_x, center_y), (sub_trunk_x - 12, center_y - 7), (sub_trunk_x - 12, center_y + 7)], fill=color_line)

                # Process children
                sub_children = item["children"]
                child_y_list = []
                sub_y_cursor = top_y

                for c in sub_children:
                    c_h = compute_height(c)
                    c_center = sub_y_cursor + ((c_h - 1) * row_height) / 2.0
                    child_y_list.append((c, c_center, sub_y_cursor))
                    sub_y_cursor += c_h * row_height

                # Draw sub-trunk line spanning children
                c_first_y = child_y_list[0][1]
                c_last_y = child_y_list[-1][1]
                draw.line([(sub_trunk_x + 35, c_first_y), (sub_trunk_x + 35, c_last_y)], fill=color_line, width=line_w)

                # Connect arrow to sub-trunk connector line
                draw.line([(sub_trunk_x, center_y), (sub_trunk_x + 35, center_y)], fill=color_line, width=line_w)

                # Recurse children
                for c_item, c_center_y, c_top_y in child_y_list:
                    draw_recursive(c_item, sub_trunk_x + 35, c_center_y, c_top_y)

        else:
            # File node
            icon_w = draw_file_icon(icon_x, center_y - 20, icon_type=item.get("icon", "file"), scale=0.95)
            text_x = icon_x + icon_w + 15
            draw.text((text_x, center_y - 15), item["name"], fill=color_text, font=font_file)

    # Execute drawing for main nodes
    for item, center_y, top_y in main_branch_y_list:
        draw_recursive(item, main_trunk_x, center_y, top_y)

    # Caption at bottom
    caption = "Figure 11. Tree-structured view of the files for the proposed eye tracking system."
    bbox = draw.textbbox((0, 0), caption, font=font_caption)
    cap_w = bbox[2] - bbox[0]
    cap_x = (width - cap_w) // 2
    cap_y = y_cursor + 40
    draw.text((cap_x, cap_y), caption, fill=(40, 40, 40), font=font_caption)

    # Crop to content height
    final_h = int(cap_y + 100)
    img_cropped = img.crop((0, 0, width, final_h))
    img_cropped.save(output_path, "PNG")
    print(f"Refined flowchart saved to {output_path} with height {final_h}px")

if __name__ == "__main__":
    draw_tree_flowchart()
