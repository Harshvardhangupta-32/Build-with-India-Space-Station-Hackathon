# get_pages.py
def get_all_pages():
    return [
        "Home",
        "Image Detection", 
        "Video Detection",
        "Multiple Images",
        "Webcam Detection",
        "Eco Mode",
        "DeepSort"
    ]

# Page configurations with emojis and descriptions
def get_page_config():
    return {
        "Home": {
            "icon": "🏠",
            "description": "Main dashboard and overview",
            "category": "main"
        },
        "Image Detection": {
            "icon": "📸",
            "description": "Single image object detection",
            "category": "detection"
        },
        "Video Detection": {
            "icon": "🎥",
            "description": "Video file processing and analysis",
            "category": "detection"
        },
        "Multiple Images": {
            "icon": "🖼️",
            "description": "Batch image processing",
            "category": "detection"
        },
        "Webcam Detection": {
            "icon": "📹",
            "description": "Real-time webcam detection",
            "category": "detection"
        },
        "Eco Mode": {
            "icon": "🌱",
            "description": "Energy-efficient detection mode",
            "category": "optimization"
        },
        "DeepSort": {
            "icon": "🎯",
            "description": "Advanced object tracking with DeepSort",
            "category": "tracking"
        }
    }

# Get page info by name
def get_page_info(page_name):
    config = get_page_config()
    return config.get(page_name, {"icon": "📄", "description": "Page description", "category": "general"})

# Get pages by category
def get_pages_by_category():
    config = get_page_config()
    categories = {}
    
    for page, info in config.items():
        category = info.get("category", "general")
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "name": page,
            "icon": info["icon"],
            "description": info["description"]
        })
    
    return categories

# Get formatted page list for navigation
def get_formatted_pages():
    pages = get_all_pages()
    config = get_page_config()
    
    formatted_pages = []
    for i, page in enumerate(pages, 1):
        page_info = config.get(page, {"icon": "📄", "description": ""})
        formatted_pages.append({
            "number": f"{i:02d}",
            "name": page,
            "icon": page_info["icon"],
            "description": page_info["description"],
            "display_name": f"{i:02d} {page_info['icon']} {page}"
        })
    
    return formatted_pages