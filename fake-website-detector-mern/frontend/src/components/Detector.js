import React, { useState } from 'react';

function Detector() {
    const [url, setUrl] = useState('');
    const [result, setResult] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);
        setError('');
        setResult('');

        try {
            const response = await fetch('http://localhost:5001/api/detect', { // Ensure this matches your backend port
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResult(data.result);
        } catch (e) {
            console.error("Error submitting URL:", e);
            setError(e.message || 'Failed to get detection result.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div>
            <h1>Fake Website Detector</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="Enter website URL"
                    required
                />
                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Detecting...' : 'Detect'}
                </button>
            </form>
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
            {result && <p>Detection Result: <strong>{result}</strong></p>}
        </div>
    );
}

export default Detector;