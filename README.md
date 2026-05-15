# Automated Traffic Analysis System

This project is an automated vehicle counting and speed estimation tool designed specifically for **CE3163 Fundamentals of Transportation Engineering - Assignment 03: Traffic Data Collection Using Automated Video Analysis**. 

It uses advanced Artificial Intelligence (YOLOv8) to track vehicles from video footage, count them by class (e.g., car, bus, truck), and calculate their real-world speeds using perspective-corrected speed traps.

---

## 🚀 How to Run the Program (For Non-Technical Users)

You don't need to be a programmer to use this! Just follow these simple steps to run the analysis on your video.

### Step 1: Prepare the Video
1. Place your traffic video file inside the project folder.
2. By default, the script looks for a video file named `traffic_video.mp4` (or whatever name is specified in the code). Make sure your video matches this name or update the `VIDEO_PATH` in the code if you know how.

### Step 2: Open the Command Line
1. Open the folder containing this project in your file explorer.
2. Click on the address bar at the top of the folder window, type `cmd`, and press **Enter**. This will open a black command prompt window directly in your project folder.

### Step 3: Run the Script
In the black command prompt window, simply copy and paste the following command and press **Enter**:

```cmd
.\.venv\Scripts\python.exe vehicle_counter_dual_lane.py
```

### Step 4: Monitor the Progress
- The script is set to run in **"headless"** mode by default. This means it won't open a video player window to show you the live tracking (which makes it run much faster!).
- You will see text in the console updating you on the progress every few hundred frames.
- **To stop the process early and save the current results**, press **`Ctrl + C`** on your keyboard. 

### Step 5: View the Results
Once the script finishes (or after you press `Ctrl + C`), it will automatically generate your reports in the same folder:
1. 📊 **`traffic_report.xlsx`**: A comprehensive Excel report containing all your data, charts, and methodology details (perfect for copying into your assignment report).
2. 🎬 **`output_counted.mp4`**: An annotated video showing the bounding boxes, tracking IDs, live speeds, and counting lines (if enabled).
3. 📸 **`gcp_benchmark.jpg`**: An image showing the exact locations of the Ground Control Points (GCPs) and speed traps used for the analysis.

---

## 🎯 Alignment with CE3163 Assignment Requirements

This software was custom-built to fulfill and exceed the requirements of the **CE3163 Assignment 03**:

### 1. Automated Vehicle Detection
- **Requirement:** Apply automated vehicle detection to extract traffic parameters.
- **Implementation:** Utilizes state-of-the-art YOLOv8 object detection paired with the BotSORT tracking algorithm. This ensures robust vehicle classification (car, bus, truck, motorcycle, etc.) and tracks individual vehicles across frames without double-counting.

### 2. Traffic Parameter Extraction (Counting)
- **Requirement:** Extract traffic parameters (Volume/Flow).
- **Implementation:** Establishes distinct counting lines (`LINE_UP` and `LINE_DOWN`) for individual lanes. Vehicles are logged exactly when they cross these thresholds, providing precise directional volume counts.

### 3. Accurate Speed Estimation (The Two-Trap Method)
- **Requirement:** Extract speed parameters and evaluate the accuracy of the method.
- **Implementation:** Instead of unreliable bounding-box travel distances, this system uses a rigorous **Dual-Line Speed Trap** per lane. 
  - The system utilizes physically measured Ground Control Points (GCPs) to calculate a **Perspective-Corrected Hyperbolic Model**. 
  - This solves the issue of perspective distortion (where pixels further from the camera represent more physical distance than pixels closer to the camera). 
  - By tracking the exact time a vehicle's center point crosses Trap Line A and Trap Line B, it calculates highly accurate real-world speeds (km/h).

### 4. Comprehensive Reporting
- **Requirement:** Critical evaluation and presentation of traffic data.
- **Implementation:** The system auto-generates `traffic_report.xlsx`, featuring 10+ sheets designed directly for the assignment:
  - **Summary & Composition:** Total counts and percentage breakdowns by vehicle class.
  - **Camera & Site Info:** Documentation of the methodology, camera height, pitch, GCP coordinates, and the speed calculation formula used.
  - **Speed Distributions:** Statistical breakdowns (Mean, Std Dev), binned histograms (e.g., 0-20 km/h, 20-40 km/h), and visual bar charts.
  - **Time-Series Flow Rates:** 1-minute, 5-minute, and 10-minute flow rate calculations (vehicles/hour) for capacity analysis.

---

## ⚙️ Advanced Settings

If you need to tweak the system, you can open `vehicle_counter_dual_lane.py` in a text editor (like Notepad) and change these simple variables at the top:

- `SHOW_PREVIEW = True` : Change this to `True` if you want to watch the AI track the cars in real-time (note: this will slow down the processing).
- `OUTPUT_VIDEO = "output_counted.mp4"` : Set this to `None` if you don't want to save the video, which will save a lot of hard drive space and processing time.
