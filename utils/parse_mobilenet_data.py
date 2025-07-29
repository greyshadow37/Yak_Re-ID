'''Purpose:
- This script processes XML annotation files (e.g., from CVAT) to extract bounding box coordinates for yaks in images, crops the corresponding regions, and organizes them into directories based on sanitized class labels. It prepares data specifically for training or evaluating a MobileNet-based feature extractor for yak re-identification.
Key Functionality:
- Parse XML Annotations: Reads multiple XML files containing image annotations with bounding box coordinates and labels.
- Frame Offset Calculation: Determines the minimum frame number across annotation files to normalize frame indices.
- Label Sanitization: Normalizes labels to a consistent format (e.g., yak1, yak2) using the sanitize_label function.
- Image Cropping: Crops image regions based on bounding box coordinates (xtl, ytl, xbr, ybr) from the XML.
- Directory Organization: Saves cropped images into subdirectories named by sanitized class IDs (e.g., yak1, yak2) in the output directory.
- Error Handling: Skips missing images or invalid bounding boxes, with verbose logging for debugging.
- Configuration: Uses predefined paths for input annotations, images, and output directories, with support for JPG format.'''
import os
import xml.etree.ElementTree as ET
from PIL import Image

# --- CONFIGURATION ---
ANNOTATION_FILES = [
    r"D:\Yak-Identification\annotations\annotations_feature_extractor\annotations1.xml",
    r"D:\Yak-Identification\annotations\annotations_feature_extractor\annotations2.xml"
]
IMAGE_FOLDER = r"D:\Yak-Identification\annotations\annotations_feature_extractor\images"
OUTPUT_FOLDER = r"D:\Yak-Identification\repo\Yak_Re-ID\data-test"
IMAGE_FORMAT = "jpg"
VERBOSE = True

# --- UTILS ---
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_label(label):
    label = label.strip().lower()
    if label.startswith("yak"):
        return label
    return f"yak{label}"

def get_frame_offset(annotation_files):
    min_frame = float('inf')
    for file in annotation_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            for image_tag in root.findall(".//image"):
                frame_number = int(image_tag.get("id"))
                min_frame = min(min_frame, frame_number)
        except Exception as e:
            if VERBOSE:
                print(f"⚠️  Failed to parse {file}: {e}")
    return int(min_frame) if min_frame != float('inf') else 0

# --- MAIN LOGIC ---
def process_annotations(xml_file, frame_offset, box_counter):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Error parsing XML {xml_file}: {e}")
        return box_counter

    for image in root.findall(".//image"):
        frame_number_from_xml = int(image.get("id"))
        normalized_frame_number = frame_number_from_xml - frame_offset

        if normalized_frame_number < 0:
            continue

        image_name = f"frame_{normalized_frame_number:05d}.{IMAGE_FORMAT}"
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        if not os.path.exists(image_path):
            if VERBOSE:
                print(f"⛔ Missing image: {image_path}")
            continue

        for box in image.findall(".//box"):
            raw_label = box.get("label")
            class_id = sanitize_label(raw_label)
            class_dir = os.path.join(OUTPUT_FOLDER, class_id)
            ensure_dir(class_dir)

            try:
                xtl = int(float(box.get("xtl")))
                ytl = int(float(box.get("ytl")))
                xbr = int(float(box.get("xbr")))
                ybr = int(float(box.get("ybr")))

                with Image.open(image_path) as img:
                    width, height = img.size
                    xtl = max(0, min(xtl, width))
                    ytl = max(0, min(ytl, height))
                    xbr = max(0, min(xbr, width))
                    ybr = max(0, min(ybr, height))

                    if xbr <= xtl or ybr <= ytl:
                        if VERBOSE:
                            print(f"⚠️  Invalid box skipped: {image_name}, Box {box_counter}")
                        continue

                    cropped = img.crop((xtl, ytl, xbr, ybr))
                    save_name = f"{os.path.splitext(image_name)[0]}_box_{box_counter:05d}.{IMAGE_FORMAT}"
                    save_path = os.path.join(class_dir, save_name)
                    cropped.save(save_path)

                    if VERBOSE:
                        print(f"✅ Saved: {save_path}")

            except Exception as e:
                print(f"❌ Error processing box in {image_name}: {e}")
                continue

            box_counter += 1

    return box_counter

def main():
    ensure_dir(OUTPUT_FOLDER)
    frame_offset = get_frame_offset(ANNOTATION_FILES)

    if VERBOSE:
        print(f"🕐 Frame offset (min frame): {frame_offset}\n")

    box_counter = 0
    for xml_file in ANNOTATION_FILES:
        if VERBOSE:
            print(f"🔍 Processing: {xml_file}")
        box_counter = process_annotations(xml_file, frame_offset, box_counter)

    print(f"\n✅ DONE. All crops saved to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
