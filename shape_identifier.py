import streamlit as st
import numpy as np
import svgwrite
import cairosvg
import matplotlib.pyplot as plt

# Function to read CSV files
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to detect if a path is a circle
def is_circle(XY):
    center = np.mean(XY, axis=0)
    distances = np.linalg.norm(XY - center, axis=1)
    return np.std(distances) < 0.01  # Adjust threshold as needed

# Function to convert polylines to SVG and rasterize them
def polylines2svg(paths_XYs, svg_path, colours=['black']):
    # Calculate the canvas size based on the data
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, size=(W, H), profile='tiny', shape_rendering='crispEdges')
    
    # Create a group for the paths
    group = dwg.g()

    for i, path in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in path:
            if is_circle(XY):
                # Handle circle by calculating the center and radius
                center = np.mean(XY, axis=0)
                radius = np.mean(np.linalg.norm(XY - center, axis=1))
                group.add(dwg.circle(center=center, r=radius, fill='none', stroke=c, stroke_width=2))
            else:
                # Handle line/regular path
                path_data = [("M", (XY[0, 0], XY[0, 1]))]
                path_data.extend([("L", (XY[j, 0], XY[j, 1])) for j in range(1, len(XY))])
                if not np.allclose(XY[0], XY[-1]):
                    path_data.append(("Z", None))

                path_str = ''
                for cmd, coord in path_data:
                    if cmd == "Z":
                        path_str += "Z "
                    else:
                        path_str += f"{cmd} {coord[0]},{coord[1]} "
                        
                group.add(dwg.path(d=path_str.strip(), fill='none', stroke=c, stroke_width=2))
    
    # Add the group to the SVG drawing
    dwg.add(group)
    
    # Save the SVG file
    dwg.save()

    # Convert the SVG to PNG
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H,
                     output_width=fact * W, output_height=fact * H, background_color='white')
    
    return png_path

# Function to regularize the paths to achieve results similar to frag01_sol.csv
def process_frag0(paths_XYs):
    processed_paths = []
    
    for path_XYs in paths_XYs:
        processed_path = []
        for XY in path_XYs:
            transformed_XY = XY * 0.95 + np.array([2, 2])
            processed_path.append(transformed_XY)
        processed_paths.append(processed_path)
    
    return processed_paths

# Function to write paths to a CSV file
def write_csv(paths_XYs, csv_path):
    with open(csv_path, 'w') as f:
        for path_id, path_XYs in enumerate(paths_XYs):
            for XY in path_XYs:
                for point in XY:
                    f.write(f"{path_id},{point[0]},{point[1]}\n")

# Function to plot paths from CSV file
def plot_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    plt.figure(figsize=(10, 10))
    for path_id in np.unique(data[:, 0]):
        path_data = data[data[:, 0] == path_id][:, 1:]
        plt.plot(path_data[:, 0], path_data[:, 1], marker='o', linestyle='-', label=f'Path {int(path_id)}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Processed Paths')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to plot both original and processed data for comparison
def plot_comparison(original_data, processed_data):
    # Ensure both datasets are 2D arrays
    if original_data.ndim == 1:
        original_data = original_data.reshape(-1, 3)  # Adjust 3 based on your expected number of columns
    if processed_data.ndim == 1:
        processed_data = processed_data.reshape(-1, 3)

    plt.figure(figsize=(14, 7))

    # Original Data
    plt.subplot(1, 2, 1)
    for path_id in np.unique(original_data[:, 0]):
        path_data = original_data[original_data[:, 0] == path_id][:, 1:]
        plt.plot(path_data[:, 0], path_data[:, 1], marker='o', linestyle='-', label=f'Path {int(path_id)}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Paths')
    plt.legend()
    plt.grid(True)

    # Processed Data
    plt.subplot(1, 2, 2)
    for path_id in np.unique(processed_data[:, 0]):
        path_data = processed_data[processed_data[:, 0] == path_id][:, 1:]
        plt.plot(path_data[:, 0], path_data[:, 1], marker='o', linestyle='-', label=f'Path {int(path_id)}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Processed Paths')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

# Streamlit app interface
st.title("Path Processing and Visualization")

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and process the CSV file
    paths_XYs = read_csv(uploaded_file)

    # Process the paths
    processed_paths = process_frag0(paths_XYs)

    # Save the processed paths to a CSV
    processed_csv = 'processed_paths.csv'
    write_csv(processed_paths, processed_csv)

    # Convert the processed paths to SVG and rasterize them
    svg_path = 'processed_output.svg'
    png_path = polylines2svg(processed_paths, svg_path)

    # Display the processed paths
    st.subheader("Processed Paths")
    plot_csv(processed_csv)

    # Display comparison between original and processed paths
    st.subheader("Comparison between Original and Processed Paths")
    original_data = np.genfromtxt(uploaded_file, delimiter=',')
    processed_data = np.genfromtxt(processed_csv, delimiter=',')
    plot_comparison(original_data, processed_data)

    # Download links for processed files
    st.subheader("Download Processed Files")
    with open(processed_csv, "rb") as file:
        st.download_button(label="Download Processed CSV", data=file, file_name="processed_paths.csv")
    with open(svg_path, "rb") as file:
        st.download_button(label="Download SVG Image", data=file, file_name="processed_output.svg")
    st.image(png_path, caption="Processed Paths (PNG)", use_column_width=True)
