# Progetto Finale - High Performance Computing (15 punti)

**Titolo:** Accelerating 3D Coordinate Transformations with CUDA: A Performance Study  
**Autore:** Giuseppe Arcucci – Matricola 0120000322  
**Corso:** Machine Learning e Big Data  
**Insegnamento:** High Performance Computing  
**Anno accademico:** 2024/2025

---

## 🎯 Obiettivo del progetto

Sviluppare e analizzare una soluzione ad alte prestazioni per l'elaborazione di **trasformazioni 3D su coordinate** (rotazioni, traslazioni, scaling) mediante l'utilizzo della **programmazione parallela con CUDA**.

Il progetto confronta:
- Implementazioni seriali e parallele su **CPU**
- Implementazioni parallele su **GPU (CUDA)**

Test eseguiti su dataset di **1 milione** e **10 milioni** di punti.

---

## 🧩 Contenuto del repository

- `kernel.cu`: Codice CUDA per le trasformazioni 3D
- `CudaRuntime1.sln`: Soluzione Visual Studio
- `CudaRuntime1.vcxproj`: File progetto C++ configurato per CUDA
- `.gitignore`: Esclusione dei file temporanei
- `README.md`: Questa guida

---

## ⚙️ Requisiti

- Windows con **Visual Studio 2022** o superiore
- **CUDA Toolkit** installato (v11.x o superiore)
- **GPU NVIDIA** compatibile CUDA
- Nsight Systems (opzionale per profiling)

---

## 🚀 Come eseguire il progetto

1. Clona questo repository:
   ```bash
   git clone https://github.com/GiuseppeArcucci1995/test-repo.git
