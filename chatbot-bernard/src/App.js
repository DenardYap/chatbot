import logo from './logo.svg';
import './App.css';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition'
import { useEffect, useState } from 'react';
import  * as tf from "@tensorflow/tfjs"
import word_hash from './word_hash.json';
import {convertSpeech, getResponses} from "./helper.js"; 
const HIDDEN_SIZE = 8;

const PROBABILITY_THRESHOLD = 0.6;

const wordSet = new Set(
  ['{', "'", '`', '"', ']', '-', '/', ':', '!', ')', '@', '\\', '*', '~', '#', '(', '%', '_', '.', '^', ',', '>', '}', '[', '=', '+', '&', '?', '|', '<', '$', ';']
);

const NOT_CONFIDENT_RESPONSES = [
  "Sorry, I can't really understand you",
  "Sorry, I didn't get you",
  "Can you repeat that?",
  "Pardon?"
]
const MODEL_PATH = "https://chatbot-bernard.s3.us-east-2.amazonaws.com/model.json"
function randomIntFromInterval(min, max) { // min and max included 
  return Math.floor(Math.random() * (max - min + 1) + min)
}

function App() {
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable
  } = useSpeechRecognition();

  const [reply, setReply] = useState(null);
  const [model, setModel] = useState(null);
  const [responses, setResponses] = useState({});


  // loading our model to predict responses
  useEffect(() =>{
    async function load_model() {
      setModel(await tf.loadLayersModel(MODEL_PATH));
      console.log("Model loaded successfully.")

    }

    async function load_responses() {
      setResponses(await getResponses());
      console.log("Responses loaded successsfully");

    }
    load_model();
    load_responses();
  }, []);

  // Make a prediction everytime the user done talking
  // a.k.a the mic closed either manually or automatically
  useEffect(() => {
    if (transcript) {
      
      async function predict() {
        console.log(transcript)
        let cur_sentence = convertSpeech(word_hash, transcript)
        let result = await model.predict(cur_sentence)
        result = await result.data();
        console.log(result)
        let argMax = 0;
        let probability = result[0];
        for(let i = 1; i < result.length; i++) {
          if (result[i] > probability) {
            probability = result[i];
            argMax = i;
          }
        }
        console.log("Predicted value is", argMax, "with a probability of", probability)
        if (probability < PROBABILITY_THRESHOLD) {
          // not confidnet enough to answer
          let random_responses = response_list[randomIntFromInterval(0, response_list.length)];
          setReply(random_responses);
          playSound(random_responses.toLowerCase())
          return;
        }
        const response_list = responses[argMax];
        let random_responses = response_list[randomIntFromInterval(0, response_list.length)];
        setReply(random_responses);
        playSound(random_responses.toLowerCase())
          // play sounds
        
      } // predict()

      predict();
    }
  }, [listening])

    // a function to get rid of punctuation
    // since we can't save special characters as filename
    function getRidOfPunctuation(str) {
      let newStr = ""
      for (let i = 0; i < str.length; i++) {
        if (wordSet.has(str[i])) {
          continue; // skip this char
        }

        newStr += str[i];
      }
      return newStr
    }
    function playSound(mp3Name) {

      mp3Name = getRidOfPunctuation(mp3Name);
      console.log(mp3Name)
      new Audio("/sound/" + mp3Name + ".mp3").play();
    }
  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }
  if (!isMicrophoneAvailable) {
    // Render some fallback content
  }
  // TODO: Choose whether type or chat

  return (
    <div className="flex flex-col justify-evenly ">
      <div className='flex justify-center pt-5'> 

        <div className='flex justify-center text-center p-5 bg-red-400 rounded-md border-4 border-black bold text-3xl w-fit'>

          <p>Microphone: {listening ? 'on' : 'off'}</p>
        </div>
      </div>

        <div className='flex flex-row items-center justify-center h-full  '>

          <button className='p-3 rounded bg-red-400 border-black border-2 bold text-2xl w-fit m-5' onClick={SpeechRecognition.startListening}>Start</button>
          <button className='p-3 rounded bg-red-400 border-black border-2 bold text-2xl w-fit m-5'  onClick={SpeechRecognition.stopListening}>Stop</button>
          <button className='p-3 rounded bg-red-400 border-black border-2 bold text-2xl w-fit m-5'  onClick={resetTranscript}>Reset</button>
        </div>
      <p>{transcript}</p>
      <p>{reply}</p>
    </div>
  );
}

export default App;
