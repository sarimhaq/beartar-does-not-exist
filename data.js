const canvas = require('canvas');
const { Canvas, Image, ImageData, loadImage, createCanvas } = canvas;
const tf = require('@tensorflow/tfjs-node'); 
const fs = require('fs');

let imgTensorArray;

async function getCanvas(location) {
    const imgFile = await loadImage(location);
    const canvas = createCanvas(240, 240); //play around with dimension
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgFile, 0, 0, 340, 340, 0, 0, 240, 240);
    return ctx.canvas;
} 

async function getImgTensorTraining(image) {
    const ctxCanvas = await getCanvas(`./train-images/${image}.jpg`);
    const img = await tf.browser.fromPixels(ctxCanvas).toFloat();
    const offset = tf.scalar(127.5);
    const normalized =  img.sub(offset).div(offset);
    return await normalized.reshape([1, 240, 240, 3]);
}

exports.getTrainData = async function() {
    for (let i = 1; i <=  44; i++) { 
        const imgTensor = await getImgTensorTraining(i);
        if (imgTensorArray == null) {
            imgTensorArray = await tf.keep(imgTensor);
        } else {
            const oldArray = imgTensorArray;
            imgTensorArray = await tf.keep(oldArray.concat(imgTensor, 0));
            await oldArray.dispose();
        }
    }
    console.log(imgTensorArray.shape);
    return imgTensorArray;
}