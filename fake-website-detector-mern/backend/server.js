const express = require('express');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 5001; // Changed port to avoid conflict with React default

app.use(cors()); // Enable CORS for all routes
app.use(express.json()); // Middleware to parse JSON bodies

app.post('/api/detect', (req, res) => {
    const { url } = req.body;

    if (!url) {
        return res.status(400).json({ error: 'URL is required' });
    }

    // Assuming your detection script is Python and named 'detect_fake.py'
    // Adjust the command if your script is different (e.g., 'node', 'java', etc.)
    const pythonProcess = spawn('python', ['./ml_model_detector.py', url]);

    let dataToSend = '';
    pythonProcess.stdout.on('data', (data) => {
        dataToSend += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
        // Optionally send stderr to client, or handle specific errors
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        if (code === 0) {
            res.json({ result: dataToSend.trim() });
        } else {
            res.status(500).json({ error: 'Detection script failed', details: dataToSend.trim() });
        }
    });
});

app.listen(port, () => {
    console.log(`Backend server running on http://localhost:${port}`);
});