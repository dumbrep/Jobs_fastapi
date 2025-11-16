
const ws2 = new WebSocket("ws://127.0.0.1:8000/video");

const startButton = document.getElementById("start_button")

startButton.addEventListener("click", async () => {

    const dropDown = document.getElementById("role");
    const job_description = document.getElementById("job_description")
    const role = dropDown.value;
    const job_tp = document.getElementById("jtype");
    const description = job_description.value

    const jtype = job_tp.value

    console.log(description)


    await readPDF()
    // Connecting with WebSockets
        const ws = new WebSocket(`ws://127.0.0.1:8000/interview/${role}`);
    
        ws2.onopen = () => {
            console.log("Connected to video websokets successfully")
        }
        ws2.onmessage = (event) => {

        const description_job = document.getElementById("job_description")
        description_job.style.display = "none";
        video.src = `data:image/jpeg;base64,${event.data}`;
        startButton.style.display = "none";

    }
    ws.onopen = () => {
        console.log("Connected to WebSockets successfully!");
        startButton.style.display = "none";

    };

    // Text-to-Speech function
    async function textToSpeech(text) {
        return new Promise(async (resolve, reject) => {

            const speech = new SpeechSynthesisUtterance(text);
            speech.pitch = 1;
            speech.rate = 1;
            speech.volume = 1;

            speech.onend = () => {
                console.log("Speech finished");
                resolve();
            };

            speech.onerror = (error) => {
                console.error("Speech synthesis error:", error);
                reject(error);
            };
            window.speechSynthesis.speak(speech);
        });
    }


    async function readPDF() {

        const fileInput = document.getElementById("resume");
        if (!fileInput.files.length) {
            console.log("Please upload the file");
            return;
        }

        const file = fileInput.files[0];

        const reader = new FileReader()

        reader.onload = async function () {
            try {
                const pdfData = new Uint8Array(this.result);

                const pdfDoc = await pdfjsLib.getDocument({ data: pdfData }).promise;

                let pdfText = " "

                for (let i = 1; i <= pdfDoc.numPages; i++) {
                    const page = await pdfDoc.getPage(i)
                    const textContent = await page.getTextContent();
                    const pageText = textContent.items.map(item => item.str).join(' ');
                    pdfText += pageText + '\n';
                }

                console.log(pdfText)
                await fetch('http://127.0.0.1:8000/resume', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        resume_dt: pdfText,
                        job_description: description,
                        jobType: jtype
                    })
                });




            } catch (error) {
                console.log(error)
            }

        }

        reader.readAsArrayBuffer(file);
        reader.onerror = () => {
            console.log("Error while reading the pdf")
        }
    }


    // Variables
    let index = 0;
    var questions = [];
    let answers = [];
    let responses = [];
    let responseType = 1;  // 1: Expecting Question, 0: Expecting Response



    // Speech Recognition Setup
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;  // Stop after one result
    recognition.interimResults = false;  // Get final result only
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        console.log("Recognized:", transcript);

        answers.push(transcript);

        // Send answer to WebSocket
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(transcript);
        } else {
            console.warn("WebSocket is not open. Could not send answer.");
        }
        responseType = 0;
        recognition.stop()
    };

    recognition.onerror = (event) => {
        console.error("Speech Recognition Error:", event.error);
    };

    recognition.onend = () => {
        console.log("Speech recognition ended.");
    };

    // Buttons

    const startListening = document.getElementById("start_listening");
    const video = document.getElementById("video")



    // Event Listeners

    startListening.addEventListener("click", async () => {
        console.log("Starting speech recognition...");
        recognition.start();
    });


    //await readPDF()
    ws.onmessage = async (event) => {
        const receivedData = event.data.trim();

        if (responseType === 1) {
            // Receiving a question

            questions.push(receivedData);
            console.log("Question:", questions[questions.length - 1]);

            responseType = 0;
            await textToSpeech(receivedData)
            startListening.click()


        } else {
            // Receiving a response
            console.log("Response:", receivedData);
            responses.push(receivedData)

            responseType = 1;

            await textToSpeech(receivedData)
        }
    };

})


