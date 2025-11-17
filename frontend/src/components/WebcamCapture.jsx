// React hooks - special functions for React components
import React, { useRef, useEffect, useCallback } from "react";

// Library for easy webcam access
import Webcam from "react-webcam";

// Our API function to send images to backend
import { predictGesture } from "../utils/api";

// Component that captures webcam and sends frames to AI
const WebcamCapture = ({ onPrediction, isCapturing, setIsCapturing }) => {
  // REFS - References to HTML elements or values that don't trigger re-render
  // webcamRef: Direct access to webcam component (like a pointer in C++)
  const webcamRef = useRef(null);

  // intervalRef: Stores the timer ID so we can stop it later
  const intervalRef = useRef(null);

  // CAPTURE FUNCTION - Takes a photo and sends to AI
  // useCallback: Makes sure this function doesn't get recreated on every render
  const capture = useCallback(async () => {
    // Only run if webcam exists AND we're currently capturing
    if (webcamRef.current && isCapturing) {
      // Take a screenshot from webcam (returns base64 image string)
      const imageSrc = webcamRef.current.getScreenshot();

      if (imageSrc) {
        try {
          // Send image to backend API
          // await = wait for response before continuing (async/await)
          const result = await predictGesture(imageSrc);

          // Call parent function to update prediction on screen
          onPrediction(result.prediction, result.confidence);
        } catch (error) {
          // If something goes wrong, log it and show error
          console.error("Prediction error:", error);
          onPrediction("Error", 0);
        }
      }
    }
  }, [isCapturing, onPrediction]); // Re-create function if these change

  // EFFECT - Runs when isCapturing changes
  // Think: "Watch this variable, and run code when it changes"
  useEffect(() => {
    if (isCapturing) {
      // Start capturing: Run capture() every 300 milliseconds (0.3 seconds)
      // setInterval = "Do this repeatedly every X milliseconds"
      intervalRef.current = setInterval(capture, 300);
    } else {
      // Stop capturing: Clear the interval if it exists
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    // CLEANUP - Runs when component unmounts or before next effect
    // Important: Stop interval when component is removed from screen
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isCapturing, capture]); // Run this effect when these variables change

  // VIDEO SETTINGS - Configure webcam
  const videoConstraints = {
    width: 640, // Video width in pixels
    height: 480, // Video height in pixels
    facingMode: "user", // Use front camera (user-facing)
  };

  // RENDER - What shows on screen
  return (
    <div className="relative">
      {" "}
      {/* relative = position for absolute children */}
      {/* Webcam container */}
      <div className="overflow-hidden bg-gray-900 rounded-lg">
        <Webcam
          ref={webcamRef} // Connect to our ref
          audio={false} // Don't capture audio
          screenshotFormat="image/jpeg" // Format for screenshots
          videoConstraints={videoConstraints} // Apply our settings
          className="w-full h-auto" // Full width, auto height
        />
      </div>
      {/* Recording indicator - only shows when isCapturing is true */}
      {isCapturing && (
        <div className="absolute top-4 right-4">
          {" "}
          {/* Position in top-right */}
          <div className="flex items-center px-3 py-2 space-x-2 text-white bg-red-500 rounded-full">
            {/* Pulsing red dot */}
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="text-sm font-semibold">Recording</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Export component
export default WebcamCapture;
