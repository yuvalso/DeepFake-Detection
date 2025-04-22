const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  resultDiv.innerHTML = "⏳ Checking video... please wait.";

  try {
    const response = await fetch("/Detect", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    resultDiv.innerHTML = `
      <p><strong>Result:</strong> ${data.output}</p>
      <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
    `;
  } catch (err) {
    console.error("Error:", err);
    resultDiv.innerHTML = "❌ Something went wrong.";
  }
});
