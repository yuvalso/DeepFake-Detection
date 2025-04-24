const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const spinner = document.getElementById("spinner");

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");

dropArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  dropArea.querySelector("p").textContent = fileInput.files[0].name;
});

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
  fileInput.files = e.dataTransfer.files;
  dropArea.querySelector("p").textContent = fileInput.files[0].name;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  resultDiv.innerHTML = "";
  spinner.classList.remove("hidden");

  try {
    const response = await fetch("/Detect", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    spinner.classList.add("hidden");

    const resultClass = data.output === "REAL" ? "result-real" : "result-fake";
    resultDiv.className = resultClass;
    // resultDiv.innerHTML = `
    //   <p><strong>Result:</strong> ${data.output}</p>
    //   <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
    // `;

    resultDiv.innerHTML = `
    <p><strong>Result:</strong> ${data.output}</p>
    <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
    `;

    const camFrameDiv = document.getElementById("cam-frame");
    if (data.cam_image) {
      camFrameDiv.innerHTML = `
        <img src="${data.cam_image}?t=${new Date().getTime()}" alt="Attention Map" style="max-width: 400px; border: 2px solid black;"/>
      `;
    } else {
      camFrameDiv.innerHTML = ""; // If no image, clear the area
    }



  } catch (err) {
    console.error("Error:", err);
    spinner.classList.add("hidden");
    resultDiv.className = "";
    resultDiv.innerHTML = "An error occurred during detection.";
  }
});
