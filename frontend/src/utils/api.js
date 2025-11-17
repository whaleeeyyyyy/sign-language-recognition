import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export const predictGesture = async (imageBase64) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      image: imageBase64,
    });

    return {
      prediction: response.data.prediction,
      confidence: response.data.confidence,
    };
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};
