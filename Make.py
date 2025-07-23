from bing_image_downloader import downloader

# Download 200 bear images into 'bears_dataset' folder
downloader.download("elephant", limit=300, output_dir='bears_dataset', adult_filter_off=True, force_replace=False, timeout=60)
