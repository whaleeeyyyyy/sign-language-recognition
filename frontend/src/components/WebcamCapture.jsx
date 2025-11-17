import React, { useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import { predictGesture } from "../utils/api";

const WebcamCapture = ({ onPrediction, isCapturing, setIsCapturing }) => {
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);

  const capture = useCallback(async () => {
    if (webcamRef.current && isCapturing) {
      const imageSrc = webcamRef.current.getScreenshot();

      if (imageSrc) {
        try {
          const result = await predictGesture(imageSrc);
          onPrediction(result.prediction, result.confidence);
        } catch (error) {
          console.error("Prediction error:", error);
          onPrediction("Error", 0);
        }
      }
    }
  }, [isCapturing, onPrediction]);

  useEffect(() => {
    if (isCapturing) {
      intervalRef.current = setInterval(capture, 300);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isCapturing, capture]);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  return (
    <div className="relative">
      <div className="rounded-lg overflow-hidden bg-gray-900">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="w-full h-auto"
        />
      </div>

      {isCapturing && (
        <div className="absolute top-4 right-4">
          <div className="flex items-center space-x-2 bg-red-500 text-white px-3 py-2 rounded-full">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="text-sm font-semibold">Recording</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamCapture;
