import './App.css';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition'
import { useEffect, useState } from 'react';
import  * as tf from "@tensorflow/tfjs"
import word_hash from './word_hash.json';
import {convertSpeech, getResponses} from "./helper.js"; 
import { IoMicCircleOutline, IoMicOffCircleOutline } from "react-icons/io5";
import animation0 from "./animation/animation0.png"
import animation1 from "./animation/animation1.png"
import animation2 from "./animation/animation2.png"
import animation3 from "./animation/animation3.png"
import animation4 from "./animation/animation4.png"
const HIDDEN_SIZE = 8;

const ANIMATION_LIST = [
  animation0,
  animation1,
  animation2,
  animation3,
  animation4
]
const PROBABILITY_THRESHOLD = 0.75;

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
  const [currentAvatar, setCurrentAvatar] = useState(ANIMATION_LIST[0])
  const [currentAvatarIndex, setCurrentAvatarIndex] = useState(0);
  const [currentMedia, setCurrentMedia] = useState(null);
  const [currentMediaPaused, setCurrentMediaPaused] = useState(true);
  const [micOpen, setMicOpen] = useState(false);
  let interval; // initialize our interval object
  // play animation
  useEffect(() => {
    if (currentMedia) {

      interval = setInterval(()=>{
        
        if (currentMedia.paused){
          setCurrentMediaPaused(true);
        }
        setCurrentAvatarIndex(prevState => (prevState + 1)%(ANIMATION_LIST.length));
      }, 150)
  
      return () => {
        clearInterval(interval);
      }
    }
  }, [currentMediaPaused])

  // effect to clean up 
  useEffect(() => {

    if (currentMedia){ 
      if (currentMedia.paused) {
        // media is paused
        clearInterval(interval);
        setCurrentAvatar(ANIMATION_LIST[0]);
        setCurrentAvatarIndex(0);
      }
    }
  }, [currentMediaPaused])

  useEffect(() => {
    setCurrentAvatar(ANIMATION_LIST[currentAvatarIndex]);
  }, [currentAvatarIndex])

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
    if (!listening){
      setMicOpen(false);

    }
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
          let random_responses = NOT_CONFIDENT_RESPONSES[randomIntFromInterval(0, NOT_CONFIDENT_RESPONSES.length - 1)];
          setReply(random_responses);
          playSound(random_responses.toLowerCase())
          return;
        }
        const response_list = responses[argMax];
        let random_responses = response_list[randomIntFromInterval(0, response_list.length - 1)];
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

    useEffect(() => {
      if (currentMedia){

        currentMedia.play();
        setCurrentMediaPaused(false);
      }
    }, [currentMedia])

    function playSound(mp3Name) {

      mp3Name = getRidOfPunctuation(mp3Name);
      console.log(mp3Name)
      setCurrentMedia(new Audio("/sound/" + mp3Name + ".mp3"))
    }

  function toggleMic() {
    console.log("toggling")
    if (micOpen) {
      setMicOpen(false);
      SpeechRecognition.stopListening();
    } else{
      setMicOpen(true);
      SpeechRecognition.startListening();

    }

  }
  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }
  if (!isMicrophoneAvailable) {
    // Render some fallback content
  }
  // TODO: Choose whether type or chat

  
  return (
    <div className="flex flex-col justify-evenly h-screen items-center">
      <div className='absolute left-2 top-2'>

      <div className=' text-[0.75em] underline'>Bernard Chatbot 1.0</div>
        <div className=' text-[0.6em]'>Developed and designed by Bernard Yap</div>
      </div>
      <div className=''>

        <div className='flex justify-center items-center pt-5 relative  '>
        <img src={currentAvatar}width={200} height={200}></img>
          {
            currentMediaPaused ? 
            <></> 
            :
            <div class=" talk-bubble tri-right left-in">
            <div class="talktext">
              <p>{reply}</p>
            </div>
          </div>
            
          }

        </div>
        <div className='flex flex-row items-center justify-center mt-[1em]'>

          <button className='' onClick={toggleMic}>
            {micOpen ? 
            <IoMicCircleOutline className='text-8xl '></IoMicCircleOutline>
            :
            <IoMicOffCircleOutline className='text-8xl '  color='red'></IoMicOffCircleOutline>}
            </button>
          </div>

      </div>

      <div className='bg-slate-200 border border-black rounded-lg mx-[2em] p-3 w-[50%]'>
            
          <p> <b>You said:</b> {transcript}</p>
      </div>

    </div>
  );
}

export default App;
