const axios = require('axios');

// Test both streaming and non-streaming endpoints
async function testApi() {
    console.log('\nNon-streaming response:');
    try {
        const response = await axios.post(
            'http://127.0.0.1:6000/predict',
            { prompt: 'What is customer churn?' },
            {
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );

        console.log(response.data.response.replace(/\n/g, ''));
    } catch (error) {
        console.error('Error in non-streaming request:', error.response?.data || error.message);
    }

    console.log('\nStreaming response:');
    try {
        const response = await axios.post(
            'http://127.0.0.1:6000/predict/stream',
            { prompt: 'What is customer churn?' },
            {
                headers: {
                    'Content-Type': 'application/json'
                },
                responseType: 'stream'
            }
        );

        let responseText = '';
        
        // Handle streaming response
        response.data.on('data', (chunk) => {
            const text = chunk.toString();
            const lines = text.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const chunk = JSON.parse(line);
                    if (chunk.token) {
                        process.stdout.write(chunk.token.replace(/\n/g, ''));
                        responseText += chunk.token;
                    } else if (chunk.done) {
                        console.log('\nStream complete');
                    } else if (chunk.error) {
                        console.error('\nError:', chunk.error);
                    }
                } catch (e) {
                    console.error('\nError parsing chunk:', e);
                }
            }
        });

        response.data.on('end', () => {
            console.log('\nStream completed');
        });

        response.data.on('error', (error) => {
            console.error('Stream error:', error);
        });
    } catch (error) {
        console.error('Error in streaming request:', error.response?.data || error.message);
    }

    console.log('\nTest complete!');
}

// Run the test
testApi();
