import React, { useState } from "react";
import Header from "./components/Header";
import WebcamCapture from "./components/WebcamCapture";
import PredictionDisplay from "./components/PredictionDisplay";

function App() {
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [isCapturing, setIsCapturing] = useState(false);
  const [sentence, setSentence] = useState("");

  const handlePrediction = (pred, conf) => {
    setPrediction(pred);
    setConfidence(conf);
  };

  const handleAddToSentence = () => {
    if (prediction && prediction !== "No hand detected") {
      setSentence((prev) => prev + prediction);
    }
  };

  const handleClearSentence = () => {
    setSentence("");
  };

  const handleAddSpace = () => {
    setSentence((prev) => prev + " ");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Webcam Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Camera Feed
            </h2>
            <WebcamCapture
              onPrediction={handlePrediction}
              isCapturing={isCapturing}
              setIsCapturing={setIsCapturing}
            />

            <div className="mt-4 flex gap-2">
              <button
                onClick={() => setIsCapturing(!isCapturing)}
                className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-colors ${
                  isCapturing
                    ? "bg-red-500 hover:bg-red-600 text-white"
                    : "bg-green-500 hover:bg-green-600 text-white"
                }`}
              >
                {isCapturing ? "Stop Recognition" : "Start Recognition"}
              </button>
            </div>
          </div>

          {/* Prediction Section */}
          <div className="space-y-6">
            <PredictionDisplay
              prediction={prediction}
              confidence={confidence}
              onAddToSentence={handleAddToSentence}
            />

            {/* Sentence Builder */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Sentence Builder
              </h2>
              <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg border-2 border-gray-200 mb-4">
                <p className="text-2xl text-gray-800 font-mono">
                  {sentence || "Your sentence will appear here..."}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleAddSpace}
                  className="flex-1 py-2 px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Add Space
                </button>
                <button
                  onClick={handleClearSentence}
                  className="flex-1 py-2 px-4 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
