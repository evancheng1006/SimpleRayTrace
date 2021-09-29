# SimpleRayTrace

https://github.com/evancheng1006/SimpleRayTrace

## Motivation
POV-ray, a virtual scene synthesizer based on ray tracing, does not accept camera models with distortion coefficients. Therefore I want to write a simple renderer that supports distortion coefficients. The camera models here are from OpenCV.

## Summary
This project contains two versions: Python and C++. For the cpp version, the renderer synthesizes a virtual scene using the provided functions in SimpleRayTrace.h.

To build the project, you need to run `cmake .` and then `make` within the cpp folder.
It also requires libcnpy and OpenCV to run the example.

This project does not implement the real ray-tracing algorithm. Instead, at most one reflection of a ray is computed.