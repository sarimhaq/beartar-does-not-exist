const tf = require('@tensorflow/tfjs-node'); 
const canvas = require('canvas');
const { Canvas, Image, ImageData, loadImage, createCanvas } = canvas;
const fs = require('fs');
const data = require('./data.js');

let latentSize = 15;
let epochs = 800;
let batchSize = 9;


//image 340x420

//target 340*340


function buildGenerator(latentSize) {
    const cnn = tf.sequential();
    cnn.add(tf.layers.dense(
        {units: 30 * 30 * 768, inputShape: [latentSize], activation: 'relu'}));
    cnn.add(tf.layers.reshape({targetShape: [30, 30, 768]}));
    cnn.add(tf.layers.conv2dTranspose({
        filters: 384,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    cnn.add(tf.layers.batchNormalization());
    cnn.add(tf.layers.conv2dTranspose({
        filters: 240,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    cnn.add(tf.layers.batchNormalization());
    cnn.add(tf.layers.conv2dTranspose({
        filters: 3,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'tanh',
        kernelInitializer: 'glorotNormal'
    }));
    /* cnn.add(tf.layers.conv2dTranspose({
        filters: 192,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
   cnn.add(tf.layers.batchNormalization());
    cnn.add(tf.layers.conv2dTranspose({
        filters: 96,
        kernelSize: 5,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    cnn.add(tf.layers.batchNormalization());
    cnn.add(tf.layers.conv2dTranspose({
        filters: 3,
        kernelSize: 5,
        strides: 1,
        padding: 'same',
        activation: 'tanh',
        kernelInitializer: 'glorotNormal'
    }));*/
    const latent = tf.input({shape: [latentSize]});
    const fakeImage = cnn.apply(latent);
    return tf.model({inputs: latent, outputs: fakeImage});
}

function buildDiscriminator() {
    const cnn = tf.sequential();
    cnn.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        padding: 'same',
        strides: 2,
        inputShape: [240, 240, 3]
    }));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));
    cnn.add(tf.layers.conv2d(
        {filters: 64, kernelSize: 3, padding: 'same', strides: 1}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));
    cnn.add(tf.layers.conv2d(
      {filters: 128, kernelSize: 3, padding: 'same', strides: 2}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));
    cnn.add(tf.layers.conv2d(
      {filters: 256, kernelSize: 3, padding: 'same', strides: 1}));
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
    cnn.add(tf.layers.dropout({rate: 0.3}));
    cnn.add(tf.layers.flatten());
    const image = tf.input({shape: [240, 240, 3]});
    const features = cnn.apply(image);
    const realnessScore =
      tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features);
    return tf.model({inputs: image, outputs: realnessScore});
}

function buildCombinedModel(latentSize, generator, discriminator, optimizer) {
    const latent = tf.input({shape: [latentSize]});
    let fakeImage = generator.apply(latent);
    discriminator.trainable = false;
    var fake = discriminator.apply(fakeImage);
    const combined =
          tf.model({inputs: latent, outputs: fake});
    combined.compile({
        optimizer,
        loss: 'binaryCrossentropy'
    });
    combined.summary();
    return combined;
}

const SOFT_ONE = 0.95;

async function trainDiscriminatorOneStep(
xTrain, batchStart, batchSize, latentSize, generator,
 discriminator) {
        const [x, y] = tf.tidy(() => {
            const imageBatch = xTrain.slice(batchStart, batchSize);
            let zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
            var generatedImages =
                  generator.predict(zVectors, {batchSize: batchSize});
            //generatedImages = tf.cast(generatedImages, 'int32');
            //console.log(generatedImages.print());
            //console.log("BATCH!!!!");
            //console.log(imageBatch.print());
            const x = tf.concat([imageBatch, generatedImages], 0);
            const y = tf.tidy(
                () => tf.concat(
                    [tf.ones([batchSize, 1]).mul(SOFT_ONE), tf.zeros([batchSize, 1])]));
            return [x, y];
        });
    const losses = await discriminator.trainOnBatch(x, y);
    tf.dispose([x, y]);
    return losses;
}

async function trainCombinedModelOneStep(batchSize, latentSize, combined) {
    const [noise, trick] = tf.tidy(() => {
    const zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    const trick = tf.tidy(() => tf.ones([batchSize, 1]).mul(SOFT_ONE));
    return [zVectors, trick];
  });

  const losses = await combined.trainOnBatch(
      noise, trick);
    tf.dispose([noise, trick]);
    return losses;
}

function makeMetadata(totalEpochs, currentEpoch, completed) {
  return {
    totalEpochs,
    currentEpoch,
    completed,
    lastUpdated: new Date().getTime()
  }
}

async function run() {
    const saveURL = `file://bear-generator-2`;
    const discriminator = await buildDiscriminator();
    discriminator.compile({
        optimizer: tf.train.adam(0.0002, 0.5),
        loss: 'binaryCrossentropy'
    });
    discriminator.summary();
    const generator = await buildGenerator(latentSize);
    generator.summary();
    const optimizer = tf.train.adam(0.0002, 0.5);
    const combined = await buildCombinedModel(latentSize, generator, discriminator, optimizer);
    let xTrain = await data.getTrainData();
    console.log("Here's the shape of xTrain>>>" + xTrain.shape);
    await generator.save(saveURL);
    let numTensors;
    let step = 0;
    for (let epoch = 0; epoch < epochs; ++epoch) {
        const tBatchBegin = tf.util.now();
        const numBatches = Math.ceil(xTrain.shape[0] / batchSize);
        for (let batch = 0; batch < numBatches; ++batch) {
            const actualBatchSize = (batch + 1) * batchSize >= xTrain.shape[0] ? 
                  (xTrain.shape[0] - batch * batchSize) : batchSize;
            console.log('after line 174');
            const dLoss = await trainDiscriminatorOneStep(xTrain, batch * batchSize, actualBatchSize, latentSize, generator, discriminator);
            const gLoss = await trainCombinedModelOneStep(2 * actualBatchSize, latentSize, combined);
            console.log(`epoch ${epoch + 1}/${epochs} batch ${batch + 1}/${
              numBatches}: ` +
                        `dLoss = ${dLoss.toFixed(6)}, gLoss = ${gLoss.toFixed(6)}`);
            if (numTensors == null) {
                numTensors = tf.memory().numTensors;
            } else {
                tf.util.assert(
                    tf.memory().numTensors === numTensors,
                    `Leaked ${tf.memory().numTensors - numTensors} tensors`);
            }
        }
        await generator.save(saveURL);
        console.log(
        `epoch ${epoch + 1} elapsed time: ` +
        `${((tf.util.now() - tBatchBegin) / 1e3).toFixed(1)} s`);
        console.log(`Saved generator model to: ${saveURL}\n`);
    }
}

async function predict() {
    let ganModel;
    ganModel = await tf.loadLayersModel('file://bear-generator-2/model.json');
    let seedTensor = await tf.randomUniform([1, latentSize], -1, 1);
    const generatedImages = await ganModel.predict(seedTensor);
    const offset = tf.scalar(127.5);
    const unNormalized = await generatedImages.mul(offset).add(offset).squeeze().toInt();
    console.log(unNormalized.print()); 
    tf.node.encodeJpeg(unNormalized).then((f) => { 
        fs.writeFileSync("simple-1.jpg", f); 
        console.log("Basic JPG 'simple.jpg' written");
    });
    //await tf.browser.toPixels(unNormalized, canvas);
   // const buffer = await canvas.toBuffer('image/png');
   // fs.writeFileSync('./test.png', buffer);
    
   // console.log(unNormalized.print()); 
}

run();

//predict();