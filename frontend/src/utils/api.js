// Axios - library for making HTTP requests (talking to backend)
import axios from "axios";

// BACKEND URL - Where our AI server is running
// localhost = this computer, 8000 = port number
const API_BASE_URL = "http://localhost:8000";

/**
 * Send image to backend and get prediction
 *
 * @param {string} imageBase64 - Base64 encoded image string
 * @returns {Object} - { prediction: 'A', confidence: 0.95 }
 */
export const predictGesture = async (imageBase64) => {
  try {
    // POST request = Sending data to server
    // Like sending a letter to the backend
    const response = await axios.post(
      `${API_BASE_URL}/predict`, // URL endpoint
      { image: imageBase64 } // Data we're sending (the image)
    );

    // Extract prediction and confidence from response
    // response.data = The JSON object backend sent back
    return {
      prediction: response.data.prediction, // e.g., "A"
      confidence: response.data.confidence, // e.g., 0.95
    };
  } catch (error) {
    // If request fails (backend down, network error, etc.)
    console.error("API Error:", error);
    throw error; // Pass error up to calling function
  }
};
