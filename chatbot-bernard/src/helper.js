// script to convert my speech to input for my model
import  * as tf from "@tensorflow/tfjs"
import training_data from './data.json';

const wordSet = new Set(
    ['{', '`', '"', ']', '-', '/', ':', '!', ')', '@', '\\', '*', '~', '#', '(', '%', '_', '.', '^', ',', '>', '}', '[', '=', '&', '?', '|', '<', '$', ';']
);

function isNumeric(s) {
    if (s == "1") {
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
    console.log("Cur_sentence:",cur_sentence)
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
    
    let intents_and_responses = {}
    training_data = training_data["intents"]
    // for i, obj in enumerate(training_data["intents"]):
    let keys = Object.keys(training_data);
    for (let i = 0; i < keys.length; i++){

        intents_and_responses[keys[i]] = training_data[keys[i]]["responses"];
    }
    return intents_and_responses
}

export {convertSpeech, getResponses};
