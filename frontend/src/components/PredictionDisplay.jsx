import React from "react";

const PredictionDisplay = ({ prediction, confidence, onAddToSentence }) => {
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return "text-green-600";
    if (conf >= 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  const getConfidenceBarColor = (conf) => {
    if (conf >= 0.8) return "bg-green-500";
    if (conf >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Current Prediction
      </h2>

      <div className="space-y-4">
        {/* Prediction Display */}
        <div className="text-center p-8 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg border-2 border-indigo-200">
          <p className="text-6xl font-bold text-indigo-600 mb-2">
            {prediction || "—"}
          </p>
          <p className="text-gray-600 text-sm">Detected Sign</p>
        </div>

        {/* Confidence Bar */}
        {prediction && (
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-semibold text-gray-700">
                Confidence
              </span>
              <span
                className={`text-sm font-bold ${getConfidenceColor(
                  confidence
                )}`}
              >
                {(confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full ${getConfidenceBarColor(
                  confidence
                )} transition-all duration-300`}
                style={{ width: `${confidence * 100}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Add to Sentence Button */}
        {prediction && prediction !== "No hand detected" && (
          <button
            onClick={onAddToSentence}
            className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold transition-colors"
          >
            Add "{prediction}" to Sentence
          </button>
        )}
      </div>
    </div>
  );
};

export default PredictionDisplay;
