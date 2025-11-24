import React, { useState } from 'react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [urlInput, setUrlInput] = useState(''); 
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false); 

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });

      const data = await response.json();
      const aiMessage = { role: 'ai', content: data.answer };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [...prev, { role: 'ai', content: 'Error connecting to server.' }]);
    }

    setInput('');
    setIsLoading(false);
  };

  
  const handleRetrain = async (e) => {
    e.preventDefault();
    if (!urlInput.trim()) return;

    setIsTraining(true);
    
    setMessages(prev => [...prev, { role: 'ai', content: `📚 Learning from ${urlInput}... please wait.` }]);

    try {
      const response = await fetch('http://localhost:8000/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInput }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, { role: 'ai', content: `✅ Success! I have processed that page and added it to my knowledge base.` }]);
      } else {
        setMessages(prev => [...prev, { role: 'ai', content: '❌ Failed to learn from that URL.' }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { role: 'ai', content: '❌ Error connecting to training endpoint.' }]);
    }
    setUrlInput('');
    setIsTraining(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RTX 4090 AI Assistant</h1>
      </header>

    
      <div className="training-section" style={{ padding: '10px', background: '#f0f0f0', borderBottom: '1px solid #ccc' }}>
        <form onSubmit={handleRetrain} style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
          <input
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="Paste a URL to learn from (e.g., Wikipedia)..."
            style={{ width: '60%', padding: '8px' }}
            disabled={isTraining}
          />
          <button type="submit" disabled={isTraining} style={{ padding: '8px 16px', backgroundColor: '#282c34', color: 'white', border: 'none', cursor: 'pointer' }}>
            {isTraining ? 'Learning...' : 'Learn URL'}
          </button>
        </form>
      </div>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <p>{msg.content}</p>
          </div>
        ))}
        {isLoading && (
          <div className="message ai">
            <p>Thinking...</p>
          </div>
        )}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the RTX 4090..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}

export default App;