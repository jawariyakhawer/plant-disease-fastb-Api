const API_URL = "http://127.0.0.1:8000/predict";

let selectedFile = null;


// ================= ELEMENTS =================

const input = document.getElementById("input");

const chooseBtn = document.getElementById("choose");

const dropArea = document.getElementById("drop");

const placeholder = document.getElementById("placeholder");

const preview = document.getElementById("preview");

const previewImage = document.getElementById("img");

const removeBtn = document.getElementById("remove");

const fileName = document.getElementById("fname");

const analyzeBtn = document.getElementById("analyze");

const errorBox = document.getElementById("error");

const emptyResult = document.getElementById("empty");

const result = document.getElementById("result");

const predictedClass = document.getElementById("pred");

const confidenceText = document.getElementById("conftext");

const confidenceValue = document.getElementById("conf");

const confidenceBar = document.getElementById("bar");

const statusText = document.getElementById("status");

const resultIcon = document.getElementById("icon");

const predictionList = document.getElementById("list");

const resetBtn = document.getElementById("reset");


// ================= CHOOSE IMAGE =================

chooseBtn.addEventListener("click", () => {

    input.click();

});


// ================= FILE SELECT =================

input.addEventListener("change", () => {

    const file = input.files[0];

    if (!file) {
        return;
    }

    setSelectedFile(file);

});


// ================= DRAG & DROP =================

["dragenter", "dragover"].forEach(eventName => {

    dropArea.addEventListener(eventName, event => {

        event.preventDefault();

        dropArea.style.borderColor = "#4ade80";

    });

});


["dragleave", "drop"].forEach(eventName => {

    dropArea.addEventListener(eventName, event => {

        event.preventDefault();

        dropArea.style.borderColor = "";

    });

});


dropArea.addEventListener("drop", event => {

    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    setSelectedFile(file);

});


// ================= SET FILE =================

function setSelectedFile(file) {

    if (
        file.type !== "image/jpeg" &&
        file.type !== "image/png"
    ) {

        showError(
            "Please choose a JPG, JPEG or PNG image."
        );

        return;

    }


    selectedFile = file;


    const reader = new FileReader();


    reader.onload = function(event) {

        previewImage.src = event.target.result;

        fileName.textContent = file.name;

        placeholder.classList.add("hidden");

        preview.classList.remove("hidden");

        analyzeBtn.disabled = false;

        hideError();

    };


    reader.readAsDataURL(file);

}


// ================= REMOVE IMAGE =================

removeBtn.addEventListener("click", () => {

    clearImage();

});


function clearImage() {

    selectedFile = null;

    input.value = "";

    previewImage.src = "";

    placeholder.classList.remove("hidden");

    preview.classList.add("hidden");

    analyzeBtn.disabled = true;

}


// ================= ANALYZE =================

analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        showError("Please select an image first.");

        return;

    }


    const originalText = analyzeBtn.innerHTML;


    analyzeBtn.disabled = true;

    analyzeBtn.innerHTML =
        "⏳ &nbsp; <span>Analyzing...</span>";


    hideError();


    const formData = new FormData();

    formData.append("file", selectedFile);


    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.detail ||
                data.message ||
                "Prediction failed."
            );

        }


        showPrediction(data);

    }

    catch (error) {

        console.error(error);

        showError(
            "Could not connect to FastAPI. Make sure your backend is running at http://127.0.0.1:8000"
        );

    }

    finally {

        analyzeBtn.disabled = !selectedFile;

        analyzeBtn.innerHTML = originalText;

    }

});


// ================= SHOW PREDICTION =================

function showPrediction(data) {

    emptyResult.classList.add("hidden");

    result.classList.remove("hidden");


    const confidence =
        Number(data.confidence) || 0;


    predictedClass.textContent =
        formatClassName(data.predicted_class);


    confidenceText.textContent =
        confidence.toFixed(2) + "% confidence";


    confidenceValue.textContent =
        confidence.toFixed(2) + "%";


    confidenceBar.style.width =
        Math.min(confidence, 100) + "%";


    // Status

    if (confidence >= 80) {

        statusText.textContent =
            "CONFIDENT PREDICTION";

        resultIcon.textContent = "✓";

    }

    else if (confidence >= 60) {

        statusText.textContent =
            "MODERATE CONFIDENCE";

        resultIcon.textContent = "!";

    }

    else {

        statusText.textContent =
            "LOW CONFIDENCE";

        resultIcon.textContent = "?";

    }


    // Top 5

    predictionList.innerHTML = "";


    if (data.top_5 && data.top_5.length > 0) {

        data.top_5.forEach((prediction, index) => {

            const confidence =
                Number(prediction.confidence) || 0;


            const row =
                document.createElement("div");

            row.className = "row";


            row.innerHTML = `

                <div class="meta">

                    <span class="name">
                        ${index + 1}.
                        ${escapeHTML(
                            formatClassName(
                                prediction.class
                            )
                        )}
                    </span>

                    <span class="percent">
                        ${confidence.toFixed(2)}%
                    </span>

                </div>


                <div class="mini">

                    <i
                        style="
                            width:${Math.min(
                                confidence,
                                100
                            )}%;
                        "
                    ></i>

                </div>

            `;


            predictionList.appendChild(row);

        });

    }

}


// ================= FORMAT CLASS =================

function formatClassName(className) {

    if (!className) {

        return "Unknown";

    }


    return String(className)

        .replaceAll("___", " — ")

        .replaceAll("_", " ")

        .replace(/,\s*/g, ", ")

        .replace(/\s+/g, " ")

        .trim();

}


// ================= ESCAPE HTML =================

function escapeHTML(text) {

    return text

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;")

        .replaceAll("'", "&#039;");

}


// ================= RESET =================

resetBtn.addEventListener("click", () => {

    clearImage();

    emptyResult.classList.remove("hidden");

    result.classList.add("hidden");

    confidenceBar.style.width = "0%";

    hideError();

});


// ================= ERROR =================

function showError(message) {

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");

}


function hideError() {

    errorBox.classList.add("hidden");

}