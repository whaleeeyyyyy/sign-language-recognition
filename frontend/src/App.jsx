// Import React - the library that helps us build user interfaces
import React, { useState } from "react";

// Import our custom components (like importing LEGO pieces)
import Header from "./components/Header";
import WebcamCapture from "./components/WebcamCapture";
import PredictionDisplay from "./components/PredictionDisplay";

// Main App component - this is like the "main()" function in Python
function App() {
  // STATE VARIABLES - Think: Variables that trigger re-rendering when changed
  // Like when you update a variable and the screen updates automatically

  // prediction: What letter the AI thinks you're showing
  // setPrediction: Function to update prediction
  // useState(''): Start with empty string
  const [prediction, setPrediction] = useState("");

  // confidence: How sure the AI is (0.0 to 1.0, like 0% to 100%)
  const [confidence, setConfidence] = useState(0);

  // isCapturing: Is the camera currently analyzing? (true/false)
  const [isCapturing, setIsCapturing] = useState(false);

  // sentence: The full sentence being built
  const [sentence, setSentence] = useState("");

  // HANDLER FUNCTIONS - Functions that respond to events

  // Called when AI makes a prediction
  // pred = predicted letter, conf = confidence score
  const handlePrediction = (pred, conf) => {
    setPrediction(pred); // Update what letter is shown
    setConfidence(conf); // Update confidence percentage
  };

  // Called when user clicks "Add to Sentence"
  const handleAddToSentence = () => {
    // Only add if there's a valid prediction
    if (prediction && prediction !== "No hand detected") {
      // setSentence(prev => prev + prediction)
      // Breakdown: prev = current sentence, we add prediction to end
      setSentence((prev) => prev + prediction);
    }
  };

  // Called when user clicks "Clear"
  const handleClearSentence = () => {
    setSentence(""); // Set sentence back to empty
  };

  // Called when user clicks "Add Space"
  const handleAddSpace = () => {
    setSentence((prev) => prev + " "); // Add a space character
  };

  // RETURN - What gets displayed on screen
  // JSX = HTML-like syntax in JavaScript
  return (
    // Main container - full screen height, gradient background
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header component at top */}
      <Header />

      {/* Main content area */}
      <main className="container px-4 py-8 mx-auto">
        {/* Grid layout: 1 column on mobile, 2 columns on large screens */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          {/* LEFT COLUMN: Webcam Section */}
          <div className="p-6 bg-white shadow-lg rounded-xl">
            <h2 className="mb-4 text-2xl font-bold text-gray-800">
              Camera Feed
            </h2>

            {/* Webcam component - captures and sends frames */}
            <WebcamCapture
              onPrediction={handlePrediction} // Pass function to update predictions
              isCapturing={isCapturing} // Pass current capturing state
              setIsCapturing={setIsCapturing} // Pass function to change state
            />

            {/* Start/Stop button */}
            <div className="flex gap-2 mt-4">
              <button
                // When clicked, toggle isCapturing (true ↔ false)
                onClick={() => setIsCapturing(!isCapturing)}
                // Conditional styling: red if capturing, green if not
                className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-colors ${
                  isCapturing
                    ? "bg-red-500 hover:bg-red-600 text-white"
                    : "bg-green-500 hover:bg-green-600 text-white"
                }`}
              >
                {/* Conditional text: Show different text based on state */}
                {isCapturing ? "Stop Recognition" : "Start Recognition"}
              </button>
            </div>
          </div>

          {/* RIGHT COLUMN: Predictions and Sentence Builder */}
          <div className="space-y-6">
            {/* Display current prediction */}
            <PredictionDisplay
              prediction={prediction}
              confidence={confidence}
              onAddToSentence={handleAddToSentence}
            />

            {/* Sentence builder box */}
            <div className="p-6 bg-white shadow-lg rounded-xl">
              <h2 className="mb-4 text-2xl font-bold text-gray-800">
                Sentence Builder
              </h2>

              {/* Display area for built sentence */}
              <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg border-2 border-gray-200 mb-4">
                <p className="font-mono text-2xl text-gray-800">
                  {/* Show sentence or placeholder text */}
                  {sentence || "Your sentence will appear here..."}
                </p>
              </div>

              {/* Control buttons */}
              <div className="flex gap-2">
                <button
                  onClick={handleAddSpace}
                  className="flex-1 px-4 py-2 font-semibold text-white transition-colors bg-blue-500 rounded-lg hover:bg-blue-600"
                >
                  Add Space
                </button>
                <button
                  onClick={handleClearSentence}
                  className="flex-1 px-4 py-2 font-semibold text-white transition-colors bg-gray-500 rounded-lg hover:bg-gray-600"
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

// Export so other files can use this component
export default App;
