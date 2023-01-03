// script to convert my speech to input for my model
import  * as tf from "@tensorflow/tfjs"
import training_data from './data.json';

const wordSet = new Set(
    ['{', '`', '"', ']', '-', '/', ':', '!', ')', '@', '\\', '*', '~', '#', '(', '%', '_', '.', '^', ',', '>', '}', '[', '=', '&', '?', '|', '<', '$', ';']
);

function randomIntFromInterval(min, max) { // min and max included 
    return Math.floor(Math.random() * (max - min + 1) + min)
  }

function isNumeric(s) {
    if (s === "1") {
        return false
    }
    return !isNaN(s - parseFloat(s));
}

function extract(input_string){
    let newS = ""
    for (let i = 0; i < input_string.length; i++){

        if ((isNumeric(input_string[i])) || (wordSet.has(input_string[i]))){
            continue
        }

        newS += input_string[i].toLowerCase()
    }
            
    return newS.split(" ")
}

// in preprocessing we stripped off all the numbers and 
// special characters, but the speech recognition API 
// will convert things such as 'one plus one' as '1 + 1'
// thus, we need to encode '1 + 1' back to 'one plus one' here
const SPECIAL_CHARACTER_MAP = {
    "1" : "one",
    "+" : "plus"
}
// convert speech to a bag of words
function convertSpeech(word_hash, sentence) {
    
    // convert sentence to 0 and 1 arrays 
    
    let cur_sentence = extract(sentence);
    let X = new Array(Object.keys(word_hash).length).fill(0)
    for (let i = 0; i < cur_sentence.length; i++) {

        // let unique_word = stemmer(cur_sentence[i])
        let unique_word = cur_sentence[i]
        
        if (unique_word in SPECIAL_CHARACTER_MAP) {
            unique_word = SPECIAL_CHARACTER_MAP[unique_word]
        }

        if (!(unique_word in word_hash)){
            continue
        }
        X[word_hash[unique_word]] = 1
    }
    return tf.tensor2d(X, [1, Object.keys(word_hash).length])
} 

// return a hash table where the key is the label of the intents
// and the value is the list of responses that we gonna respond with 
async function getResponses() {
    
    /**
     * {
     *   0 : {
     *      responses: [... ... ...],
     *      context_set: [31, 33, 35]
     *   }
     * }
     */
    let intents_and_responses = {}
    
    // find out the indexes of the context
    // Key: actual string of the context
    // Index: the index of the context in training_data (a.k.a the order)
    let context_indexes = {}

    training_data = training_data["intents"]
    // get the keys 
    let keys = Object.keys(training_data);

    // first find out the indexes of all the context 
    // do two passes
    
    for (let i = 0; i < keys.length; i++){
        if (!training_data[keys[i]]["context_filter"]){
            continue;
        }
        context_indexes[training_data[keys[i]]["intent"]] = keys[i]

    }

    for (let i = 0; i < keys.length; i++){
        let curContextArray = training_data[keys[i]]["context_set"];
        if (curContextArray){
            intents_and_responses[keys[i]] = {responses: [], context_set: []}
            
            for (let j = 0; j < curContextArray.length; j++) {
                intents_and_responses[keys[i]]["context_set"].push(context_indexes[curContextArray[j]])
            }
            // if i have a context_set in the current entry, I want to 
            // record it 

        } else {
            // this is not a context_set but a context_filter instead 
            intents_and_responses[keys[i]] = {responses: [], context_set: false}
            
        }
        intents_and_responses[keys[i]]["responses"] = training_data[keys[i]]["responses"];
    }
    return intents_and_responses
}

export {convertSpeech, getResponses, randomIntFromInterval};
