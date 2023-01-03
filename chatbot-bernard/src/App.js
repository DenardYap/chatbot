import './App.css';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition'
import { useEffect, useState } from 'react';
import  * as tf from "@tensorflow/tfjs"
import word_hash from './word_hash.json';
import {convertSpeech, getResponses, randomIntFromInterval} from "./helper.js"; 
import { IoMicCircleOutline, IoMicOffCircleOutline } from "react-icons/io5";
import animation0 from "./animation/animation0.png"
import animation1 from "./animation/animation1.png"
import animation2 from "./animation/animation2.png"
import animation3 from "./animation/animation3.png"
import animation4 from "./animation/animation4.png"

const style = 'color:blue; font-size:20px; font-weight: bold; -webkit-text-stroke: 1px black;'

console.log("%cCODED AND DEVELOPED BY BERNARD YAP", style)
// the order to play the animation when Bernard is talking
const ANIMATION_LIST = [
  animation0,
  animation1,
  animation2,
  animation3,
  animation4
]

// We only play a response if the confidence is at least 90%
const PROBABILITY_THRESHOLD = 0.9;
// We only play a context response if the confidence is at least 1%
const CONTEXT_PROBABILITY_THRESHOLD = 0.01; // maybe can be lower or higher

// Responses to be played when probability threshold is not reached 
const NOT_CONFIDENT_RESPONSES = [
  "Sorry, I can't really understand you",
  "Sorry, I didn't get you",
  "Can you repeat that?",
  "Pardon?"
]

// path to my model's CDN
const MODEL_PATH = "https://chatbot-bernard.s3.us-east-2.amazonaws.com/model.json"

// Punctuations to filtered out when converting speech to BOW
const wordSet = new Set(
  ['{', "'", '`', '"', ']', '-', '/', ':', '!', ')', '@', '\\', '*', '~', '#', '(', '%', '_', '.', '^', ',', '>', '}', '[', '=', '+', '&', '?', '|', '<', '$', ';']
);

function App() {
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
    isMicrophoneAvailable
  } = useSpeechRecognition();

  // The responses/reply randomly chosen after user talked 
  const [reply, setReply] = useState(null);
  // the model we used to predict
  const [model, setModel] = useState(null);
  // A hash table. Key is intent's index, value is the list of responses from data.json
  // It also has another key called 'context', where it record all the possible context and their
  // respective intents ID and list of responses. 
  const [responsesAndContext, setResponsesAndContext] = useState({})
  // A hash table. Key is context intent's index, value is the list of responses from data.json

  // the image displayed in the middle of the page
  const [currentAvatar, setCurrentAvatar] = useState(ANIMATION_LIST[0])
  // the image index, used to switch between different animation
  const [currentAvatarIndex, setCurrentAvatarIndex] = useState(0);
  // the audio object
  const [currentMedia, setCurrentMedia] = useState(null);
  // Check if the audio is paused
  const [currentMediaPaused, setCurrentMediaPaused] = useState(true);
  // Check if the mic is opened
  const [micOpen, setMicOpen] = useState(false);

  // the context we are having now
  const [currentContext, setCurrentContext] = useState(null);
  
  // interval to play animation, initialized here so we can stop it later globally
  let interval; 
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
      // TODO: When loading Load responses also load the “context_set"
      setResponsesAndContext(await getResponses());
      console.log("Responses loaded successsfully");

    }
    load_model();
    load_responses();
  }, []);

  function checkGotContext(entry) {
    if (responsesAndContext[entry]["context_set"]) {
      return false;
    } 
    return true;
  }
  // Make a prediction everytime the user done talking
  // a.k.a the mic closed either manually or automatically
  useEffect(() => {
    if (!listening){
      setMicOpen(false);

    }
    if (transcript) {
      
      /** Case 1:
       *  Made a prediciton that doesn't have context, reset context to null no matter what (even if originally null)
       *  
       *  Case 2:
       *  Current context is null, Made a prediction with context_set, then we set context to that index
       *  
       *  Case 3:
       *  Current context is set, Made a prediction with context_set, then we set context to that index
       *  
       *  Case 4:
       *  Current context is set, Made a prediction with context_filter, show that response if the 
       *  probability is higher than CONTEXT_PROBABILITY_THRESHOLD 
       * 
       */
      async function predict() {
        let cur_sentence = convertSpeech(word_hash, transcript)
        if (tf.sum(cur_sentence).dataSync()[0] == 0){

          let random_responses = NOT_CONFIDENT_RESPONSES[randomIntFromInterval(0, NOT_CONFIDENT_RESPONSES.length - 1)];
          setReply(random_responses);
          playSound(random_responses.toLowerCase())
          return;
        }
        let result = await model.predict(cur_sentence)
        result = await result.data();
        let argMax = -1;
        let probability = 0;
        let possibleResponses = [];

        // first initialize all the possible predicted entries
        // this list may contain both entries that have or don't have context
        for(let i = 0; i < result.length; i++) {
          
          if (result[i] > CONTEXT_PROBABILITY_THRESHOLD) {
            possibleResponses.push(i);
            console.log("pushed into possibleResponses of index", i, "with probability", result[i])
          }
        }
        
        // if we do not have a context going on
        // just predict the highest probability which we already did -> argMax and probability
        // else if we have a current context going on
        let inContext = false;
        if (currentContext) {
          // go through the possibleResponses list and choose the one that has a context

          // if all the possibleResponses are context, we do not want to give a response
          for (let k = 0; k < possibleResponses.length; k++) {

            // we grab the first context we found, and just set the argMax to that entry with context

            for (let m = 0; m < currentContext.length; m++) {
              if (currentContext[m] === possibleResponses[k].toString()) {

                argMax = possibleResponses[k];
                probability = result[possibleResponses[k]]
                k = possibleResponses.length;
                break;
              }
            }
          }

          // If we don't find any matching context, just choose the most confidnet non-context response
          if (argMax == -1) {
            
            // then we choose the highest non-context response
            for(let k = 0; k < possibleResponses.length; k++) {
              if (result[possibleResponses[k]] > probability && !checkGotContext(possibleResponses[k])) {
                probability = result[possibleResponses[k]];
                argMax = possibleResponses[k];
              }
              
            }
          }

          // if there isn't even 1 non-context response, we go ahead and reply not confident
          if (argMax == -1) {
            setCurrentContext(null);
            let random_responses = NOT_CONFIDENT_RESPONSES[randomIntFromInterval(0, NOT_CONFIDENT_RESPONSES.length - 1)];
            setReply(random_responses);
            playSound(random_responses.toLowerCase())
            return;
          }
        } else {
          // if currently dont have any context 
          let k = 0;

          // we strip off all the entries that have context
          // since we won't predict them anyway 
          while (k < possibleResponses.length) {
            if (checkGotContext(possibleResponses[k])) {
              possibleResponses.splice(k, 1)
            } else{
              k++;
            }

          }
          
          // if all the entries have context, the resulting list will have length of 0
          // thus, we just return not-confident response
          if (possibleResponses.length == 0) {
            setCurrentContext(null);
            let random_responses = NOT_CONFIDENT_RESPONSES[randomIntFromInterval(0, NOT_CONFIDENT_RESPONSES.length - 1)];
            setReply(random_responses);
            playSound(random_responses.toLowerCase())
            return;
          } else {
            // find out the argMax and probaiblity
            probability = result[possibleResponses[0]];
            argMax = possibleResponses[0];
            for(let i = 0; i < possibleResponses.length; i++) {
              if (result[possibleResponses[i]] > probability) {
                probability = result[possibleResponses[i]];
                argMax = possibleResponses[i];
              }
            }
          }
        }
        
        // for cases where no context and predicted label with context, we dont want to predict that
        // set context if current entry has context_set

        if (responsesAndContext[argMax]["context_set"]) {
          setCurrentContext(responsesAndContext[argMax]["context_set"]);
        } else {
          setCurrentContext(null); // back to zero
        }
        if (probability < PROBABILITY_THRESHOLD && currentContext == null) {
          // not confidnet enough to answer
          let random_responses = NOT_CONFIDENT_RESPONSES[randomIntFromInterval(0, NOT_CONFIDENT_RESPONSES.length - 1)];
          setReply(random_responses);
          playSound(random_responses.toLowerCase())
          return;
        }
        console.log("Predicted index", argMax, "with a probability", probability);
        const response_list = responsesAndContext[argMax]["responses"];
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
      setCurrentMedia(new Audio("/sound/" + mp3Name + ".mp3"))
    }

  function toggleMic() {
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

      <div className=' text-[0.75em] underline'>Bernard Chatbot 1.1</div>
        <div className=' text-[0.6em]'>Developed and designed by Bernard Yap</div>
        <div className=' text-[0.6em]'>Major update in 1.1: Added context and a few new questions/responses</div>
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
