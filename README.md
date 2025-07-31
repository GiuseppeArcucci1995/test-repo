# Final Project - High Performance Computing (15 Points)

**Title:** Accelerating 3D Coordinate Transformations with CUDA: A Performance Study  
**Author:** Giuseppe Arcucci 
**Degree Program:** M.Sc. in Machine Learning and Big Data  
**Course:** High Performance Computing  
**Academic Year:** 2024/2025

---

## 🎯 Project Goal

The goal of this project is to develop and analyze a high-performance solution for applying **3D coordinate transformations** (rotation, translation, scaling) using **parallel computing with CUDA**.

This implementation compares:
- Serial and parallel execution on **CPU**
- Fully parallelized transformation on **GPU (CUDA)**

Experiments were conducted on datasets containing **1 million** and **10 million** points.

---

## 📁 Repository Content

- `kernel.cu`: CUDA code for 3D point transformations  
- `CudaRuntime1.sln`: Visual Studio solution file  
- `CudaRuntime1.vcxproj`: Project configuration file for CUDA  
- `.gitignore`: Excludes temporary and build-related files  
- `README.md`: This file

---

## ⚙️ Requirements

- Windows with **Visual Studio 2022** or later  
- **CUDA Toolkit** installed (v11.x or later)  
- An **NVIDIA GPU** with CUDA support  
- Nsight Systems (optional for profiling)

---

## 🚀 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/GiuseppeArcucci1995/test-repo.git

