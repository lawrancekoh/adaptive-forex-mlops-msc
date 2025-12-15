# MQL5 ZMQ Integration Guide

## Overview
This guide explains how to integrate the Adaptive ML system with the FXATM Expert Advisor.

---

## Step 1: Install mql-zmq Library

1. Download the mql-zmq library:
   - GitHub: https://github.com/dingmaotu/mql-zmq
   - Or use the pre-built release

2. Copy files to your MT5 installation:
   ```
   MQL5/Include/Zmq/     ← Zmq.mqh and related files
   MQL5/Libraries/       ← libzmq.dll, libsodium.dll
   ```

3. Enable DLL imports in MT5:
   - Tools → Options → Expert Advisors
   - Check "Allow DLL imports"

---

## Step 2: Include in FXATM.mq5

Add this include statement near the top of `FXATM.mq5`:

```mql5
#include <FXATM/Managers/AdaptiveManager.mqh>
```

---

## Step 3: Initialize in OnInit()

Add to the `OnInit()` function:

```mql5
// Initialize Adaptive Manager (connects to Python ML server)
// Set enabled=true, server address, update frequency (bars)
if(!g_adaptive_manager.Initialize(true, "tcp://localhost:5555", 4))
{
    Print("Warning: Adaptive Manager failed to connect. Using static params.");
}
```

---

## Step 4: Call on Each Bar

In `OnTick()`, add after the new bar check:

```mql5
if(IsNewBar(PERIOD_M15))
{
    g_adaptive_manager.OnNewBar();
    // ... rest of trading logic
}
```

---

## Step 5: Shutdown in OnDeinit()

```mql5
void OnDeinit(const int reason)
{
    g_adaptive_manager.Shutdown();
    // ... rest of cleanup
}
```

---

## Testing

1. Start Python Inference Server:
   ```bash
   .venv/bin/python 2_PYTHON_MLOPS/src/inference_server.py
   ```

2. Attach FXATM EA to a chart in MT5

3. Check Experts tab for messages like:
   ```
   [AdaptiveManager] Updated to Regime 2 (Trending) - StepMult=2.0, LotMult=1.2
   ```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "DLL not found" | Copy libzmq.dll to MT5/Libraries |
| "Connection timeout" | Ensure Python server is running |
| "Invalid handle" | Check ZMQ library version compatibility |
