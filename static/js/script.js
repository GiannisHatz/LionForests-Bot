let flowIndex, featureNames, featureValues, categoricalFeatures, newInstance, originalInstance, meanValues, discreteFeatures, classNames, categoricalMap;

function initializeVarsFromJinja(flowIndexx, featureNamess, featureValuess, categoricalFeaturess, meanValuess, discreteFeaturess, classNamess, categoricalMapp) {
  flowIndex = flowIndexx;
  featureNames = featureNamess;
  featureValues = featureValuess;
  categoricalFeatures = categoricalFeaturess;
  newInstance = new Array(featureNames.length);
  originalInstance = new Array(featureNames.length);
  meanValues = meanValuess;
  discreteFeatures = discreteFeaturess;
  classNames = classNamess;
  categoricalMap = categoricalMapp;

}

let feature = "";
let previousBotText;

function getBotResponse() {

  $.get("/get").done(function(data) {
    let botTextFragment = document.createDocumentFragment();
    let liBot = document.createElement("li");
    liBot.setAttribute("class", "bot-msg-container");
    let imgDivBot = document.createElement("div");
    imgDivBot.setAttribute("class", "bot-img");
    let imgBot = document.createElement("img");
    imgBot.setAttribute("src", "static/images/bot.png");
    let botTxt = document.createElement("div");
    botTxt.setAttribute("class", "bot-msg card");
    if (feature != "" && (flowIndex === 6 || flowIndex === 11 || flowIndex === 16)) {
      botTxt.innerHTML = data + " " + feature + ":";
    } else {
      botTxt.innerHTML = data;
    }
    botTextFragment.appendChild(liBot);
    liBot.appendChild(imgDivBot);
    imgDivBot.appendChild(imgBot);
    liBot.appendChild(botTxt);
    botTextFragment.append(liBot);
    document.getElementById('box').appendChild(botTextFragment);

    let optionsFragment = document.createDocumentFragment();
    let optionsLI = document.createElement("li");
    let pillYes = document.createElement("button");
    let pillNo = document.createElement("button");
    optionsLI.setAttribute("class", "pill-container");
    pillYes.setAttribute("class", "pill btn btn-light btn-outline-secondary");
    pillYes.innerHTML = 'Yes';
    pillNo.setAttribute("class", "pill btn btn-light btn-outline-secondary");
    pillNo.innerHTML = 'No';
    optionsLI.style.width = botTxt.offsetWidth + "px";
    optionsLI.appendChild(pillYes);
    optionsLI.appendChild(pillNo);
    optionsFragment.appendChild(optionsLI);
    previousBotText = botTxt;
    switch (flowIndex) {
      // "Let’s predict the absence or presence of a serious killer disease in your heart. Are you in?"
      case 1:
        let optionsFragment3 = document.createDocumentFragment();
        let optionsLI3 = document.createElement("li");
        let pillReady = document.createElement("button");
        optionsLI3.setAttribute("class", "pill-container");
        pillReady.setAttribute("class", "pill btn btn-light btn-outline-secondary");
        pillReady.innerHTML = 'Ready!';
        optionsLI3.style.width = botTxt.offsetWidth + "px";
        optionsLI3.appendChild(pillReady);
        optionsFragment3.appendChild(optionsLI3);
        pillReady.addEventListener('click', ready, false);
        document.getElementById('box').appendChild(optionsFragment3);
        pillReady.scrollIntoView({
          block: 'start',
          behavior: 'smooth'
        });
        break;
        // "Would you like to inspect any feature from the rule?"
      case 4:
        pillYes.addEventListener('click', yes2, false);
        pillNo.addEventListener('click', no, false);
        document.getElementById("box").appendChild(optionsFragment);
        break;
        // "Would you like to inspect any other feature before we continue?"
      case 8:
        pillYes.innerHTML = "Inspect more"
        pillNo.innerHTML = 'Continue';
        pillYes.addEventListener('click', yes4, false);
        pillNo.addEventListener('click', no2, false);
        document.getElementById("box").appendChild(optionsFragment);
        break;
        //  "Hmm, I see. Do you want to discard your previous changes to the instance?"
      case 9:
        pillYes.innerHTML = 'Discard Changes';
        pillNo.innerHTML = 'Keep Changes';
        pillYes.addEventListener('click', yes5, false);
        pillNo.addEventListener('click', no3, false);
        document.getElementById("box").appendChild(optionsFragment);
        break;

      case 14:
        pillYes.addEventListener('click', yes3, false);
        pillNo.addEventListener('click', no, false);
        document.getElementById("box").appendChild(optionsFragment);
        break;

      case 18:
        let optionsFragment2 = document.createDocumentFragment();
        let optionsLI2 = document.createElement("li");
        let pillRestart = document.createElement("button");
        optionsLI2.setAttribute("class", "pill-container");
        pillRestart.setAttribute("class", "pill btn btn-light btn-outline-secondary");
        pillRestart.innerHTML = 'Restart session';
        optionsLI2.style.width = botTxt.offsetWidth + "px";
        optionsLI2.appendChild(pillRestart);
        optionsFragment2.appendChild(optionsLI2);
        pillRestart.addEventListener('click', restartClick, false);
        document.getElementById('box').appendChild(optionsFragment2);
        pillRestart.scrollIntoView({
          block: 'start',
          behavior: 'smooth'
        });
        break;
    }
    botTxt.scrollIntoView({
      block: 'start',
      behavior: 'smooth'
    });
  });
  flowIndex++;
}


function ready() {
  rawText = 'Ready!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  $.when($.ajax(getBotResponse())).then(function() {
    makeDropDown();
  });

}

function yes2() {
  rawText = 'Yes!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  $.when($.ajax(getBotResponse())).then(function() {
    showFeaturesDropDown();
  });

}

function yes3() {
  rawText = 'Yes!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  $.when($.ajax(getBotResponse())).then(function() {
    fetchChangesInProbabilities();
  });

}

function yes4() {
  rawText = 'Inspect!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  getBotResponse();
}

let toDiscardChanges = false;

function yes5() {
  rawText = 'Discard Changes!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  toDiscardChanges = true;
  $.when($.ajax(getBotResponse())).then(function() {
    showFeaturesDropDown();
  });
}

function no3() {
  rawText = 'Keep Changes!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  $.when($.ajax(getBotResponse())).then(function() {
    showFeaturesDropDown();
  });
}

function no2() {
  rawText = 'Continue!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });

  $.get("/setDialogFlowIndex", {
    flow_index: flowIndex + 4
  }).done(function(data) {
    flowIndex+=5;
    getBotResponse();
  });

}

function no() {
  rawText = 'Nope.'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);

  let botTextFragment = document.createDocumentFragment();
  let liBot = document.createElement("li");
  liBot.setAttribute("class", "bot-msg-container");
  let imgDivBot = document.createElement("div");
  imgDivBot.setAttribute("class", "bot-img");
  let imgBot = document.createElement("img");
  imgBot.setAttribute("src", "static/images/bot.png");
  let botTxt = document.createElement("div");
  botTxt.setAttribute("class", "bot-msg card");
  botTxt.innerHTML = 'Ok, thanks for your time, bye!';
  botTextFragment.appendChild(liBot);
  liBot.appendChild(imgDivBot);
  imgDivBot.appendChild(imgBot);
  liBot.appendChild(botTxt);
  botTextFragment.append(liBot);
  document.getElementById('box').appendChild(botTextFragment);
  botTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  let optionsFragment = document.createDocumentFragment();
  let optionsLI = document.createElement("li");
  let pillRestart = document.createElement("button");
  optionsLI.setAttribute("class", "pill-container");
  pillRestart.setAttribute("class", "pill btn btn-light btn-outline-secondary");
  pillRestart.innerHTML = 'Restart session';
  optionsLI.appendChild(pillRestart);
  optionsLI.style.width = botTxt.offsetWidth + "px";
  optionsFragment.appendChild(optionsLI);
  pillRestart.addEventListener('click', restartClick, false);
  document.getElementById('box').appendChild(optionsFragment);
  pillRestart.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
}

function restartClick() {
  $.get("/restartSession").done(function(data) {
    location.reload();
  });
}

let featuresDropDown = new Array();
let dropIndex = -1;




function makeDropDown() {

  let inputFragment = document.createDocumentFragment();
  let liWrap = document.createElement("li");
  liWrap.setAttribute("class", "input-container");
  let formWrap = document.createElement("form");

  for (let i = 0; i < featureNames.length; i++) {
    if (categoricalFeatures.includes(featureNames[i])) {
      let inGroup = document.createElement("div");
      inGroup.setAttribute("class", "form-group input-group");
      let prepend = document.createElement("div");
      prepend.setAttribute("class", "input-group-prepend");
      let label = document.createElement("label");
      label.setAttribute("class", "input-group-text");
      label.innerHTML = featureNames[i] + ":";
      label.setAttribute("for", featureNames[i]);
      let dropDown = document.createElement("select");
      dropDown.setAttribute("class", "custom-select");
      dropDown.setAttribute("name", featureNames[i]);

      for (let j = 0; j < featureValues[i].length; j++) {
        let option = document.createElement("option");
        option.setAttribute("value", featureValues[i][j]);
        option.innerHTML = featureValues[i][j];
        dropDown.appendChild(option);
      }
      formWrap.appendChild(inGroup);
      inGroup.appendChild(prepend);
      prepend.appendChild(label);
      inGroup.appendChild(dropDown);
      formWrap.appendChild(inGroup);

    } else {
      let inGroup = document.createElement("div");
      inGroup.setAttribute("class", "form-group input-group");
      let prepend = document.createElement("div");
      prepend.setAttribute("class", "input-group-prepend");
      let label = document.createElement("label");
      label.setAttribute("class", "input-group-text");
      label.innerHTML = featureNames[i] + "(" + featureValues[i][0] + "-" + featureValues[i][1] + "):";
      label.setAttribute("for", featureNames[i]);
      let input = document.createElement("input");
      input.setAttribute("type", "text");
      input.setAttribute("class", "form-control input-sm");
      input.setAttribute("name", featureNames[i]);
      if (discreteFeatures.includes(featureNames[i])) {
        input.setAttribute("value", Math.ceil(meanValues[i]).toString());
      } else {
        input.setAttribute("value", meanValues[i].toFixed(3).toString());
      }
      formWrap.appendChild(inGroup);
      inGroup.appendChild(prepend);
      prepend.appendChild(label);
      inGroup.appendChild(input);
      formWrap.appendChild(inGroup);
    }
  }
  let submitBut = document.createElement("button");
  submitBut.setAttribute("type", "button");
  submitBut.setAttribute("class", "btn btn-primary");
  submitBut.innerHTML = "Submit!";
  submitBut.addEventListener('click', instanceSubmit, false);
  formWrap.appendChild(submitBut);
  liWrap.appendChild(formWrap)
  inputFragment.appendChild(liWrap);
  document.getElementById("box").appendChild(inputFragment);
  submitBut.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
}

function instanceSubmit() {
  rawText = 'Submit!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  fetchPrediction(0);
}



function showFeaturesDropDown() {

  let inputFragment = document.createDocumentFragment();
  let liWrap = document.createElement("li");
  liWrap.setAttribute("class", "input-container");
  let formWrap = document.createElement("form");
  let inGroup = document.createElement("div");
  inGroup.setAttribute("class", "form-group input-group");
  let prepend = document.createElement("div");
  prepend.setAttribute("class", "input-group-prepend");
  let label = document.createElement("label");
  label.setAttribute("class", "input-group-text");
  label.innerHTML = "Feature:"
  label.setAttribute("for", featuresDropDown[dropIndex].id);
  formWrap.appendChild(inGroup);
  inGroup.appendChild(prepend);
  prepend.appendChild(label);
  inGroup.appendChild(featuresDropDown[dropIndex]);
  formWrap.appendChild(inGroup);
  let submitBut = document.createElement("button");
  submitBut.setAttribute("type", "button");
  submitBut.setAttribute("class", "btn btn-primary");
  submitBut.innerHTML = "Choose!";
  submitBut.addEventListener('click', changeValueOnClick, false);
  formWrap.appendChild(submitBut);
  liWrap.appendChild(formWrap)
  inputFragment.appendChild(liWrap);
  document.getElementById("box").appendChild(inputFragment);
  submitBut.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
}



function fetchPrediction(beforeChange, newValue = 0, probaClickValue = 0) {
  if (beforeChange === 0) {
    let valuesNonCategorical = document.querySelectorAll(".form-control");
    let valuesCategorical = document.querySelectorAll(".custom-select");
    for (let i = 0, element; element = valuesNonCategorical[i++];) {
      if (element.name != "") {
        newInstance[featureNames.indexOf(element.name)] = element.value;
        originalInstance[featureNames.indexOf(element.name)] = element.value;
      }
    }
    for (let i = 0, element; element = valuesCategorical[i++];) {
      let mapValue;
      Object.keys(categoricalMap).forEach(function(key) {
        if (categoricalMap[key].includes(element.options[element.selectedIndex].value)) {
            mapValue = categoricalMap[key].indexOf(element.options[element.selectedIndex].value);
        }
      });
      newInstance[featureNames.indexOf(element.name)] = mapValue;
      originalInstance[featureNames.indexOf(element.name)] = mapValue;
    }

  } else if (beforeChange === 1) {
    if (!toDiscardChanges) {
      newInstance[featureNames.indexOf(feature)] = newValue;
      console.log("KEEPING CHANGES");
      console.log(newValue);
      console.log(feature);
      console.log(newInstance)
    } else {
      originalInstance[featureNames.indexOf(feature)] = newValue;
      newInstance = originalInstance;
    }
  } else {
    originalInstance[featureNames.indexOf(feature)] = newValue;
    newInstance = originalInstance;
  }
  getBotResponse();
  $.get("/getPrediction", {
    instance: newInstance
  }).done(function(data) {
    let botTextFragment = document.createDocumentFragment();
    let liBot = document.createElement("li");
    liBot.setAttribute("class", "bot-msg-container");
    let imgDivBot = document.createElement("div");
    imgDivBot.setAttribute("class", "bot-img");
    let imgBot = document.createElement("img");
    imgBot.setAttribute("src", "static/images/bot.png");
    let botTxt = document.createElement("div");
    botTxt.setAttribute("class", "bot-msg card");
    ruleSplit = data.rule.split(" & ");
    let botHTML = "";
    for (let i = 0; i < ruleSplit.length; i++) {
      botHTML = botHTML.concat(ruleSplit[i]);
      botHTML = botHTML.concat("<br>");
    }
    botTxt.innerHTML = botHTML;
    botTextFragment.appendChild(liBot);
    liBot.appendChild(imgDivBot);
    imgDivBot.appendChild(imgBot);
    liBot.appendChild(botTxt);
    dropIndex++;
    let featuresDropDownx = document.createElement("select");
    featuresDropDownx.setAttribute("class", "custom-select");
    featuresDropDownx.setAttribute("id", "drop" + dropIndex);

    let notIncluded = new Array();
    for (let i = 0; i < featureNames.length; i++) {
      if (data.rule.includes(featureNames[i]) /*&& !categoricalFeatures.includes(featureNames[i])*/) {
        let option = document.createElement("option");
        option.setAttribute("value", featureNames[i]);
        option.innerHTML = featureNames[i];
        featuresDropDownx.appendChild(option);
      } else if (!data.rule.includes(featureNames[i])) {
        notIncluded.push(featureNames[i]);
      }
    }
    featuresDropDown.push(featuresDropDownx);
    if (beforeChange === 0) {
      let toolDiv = document.createElement('div');
      toolDiv.setAttribute("class", "tooltip-icon");
      let toolMood = document.createElement('i');
      toolMood.setAttribute("class", "las la-info-circle");
      let toolSpan = document.createElement('span');
      toolSpan.setAttribute("class", "tooltip-text");
      let string1 = "";
      if (notIncluded.length) {
        notIncluded[0] = notIncluded[0].charAt(0).toUpperCase() + notIncluded[0].slice(1);
        string1 = notIncluded.toString() + " did not influence the prediction. As long as the values of the features ";
      } else {
        string1 = "As long as the values of the features ";
      }
      let string2 = "stay inside the rule's ranges, the prediction will always be the same! ";
      let string3 = "And by the way, our model is sure by " + Math.ceil(data.accuracy * 100) + "% for the prediction.";
      toolSpan.innerHTML = string1 + string2 + string3;
      toolDiv.appendChild(toolMood);
      toolDiv.appendChild(toolSpan);
      liBot.appendChild(toolDiv);
    }
    if (beforeChange === 1) {
      let toolDiv = document.createElement('div');
      toolDiv.setAttribute("class", "tooltip-icon");
      let toolMood = document.createElement('i');
      toolMood.setAttribute("class", "las la-info-circle");
      let toolSpan = document.createElement('span');
      toolSpan.setAttribute("class", "tooltip-text");
      toolSpan.innerHTML = "Notice the change in the ranges our rule gives for " + feature + " after you changed its value!";
      toolDiv.appendChild(toolMood);
      toolDiv.appendChild(toolSpan);
      liBot.appendChild(toolDiv);
    }
    if (beforeChange === 2) {
      let toolDiv = document.createElement('div');
      toolDiv.setAttribute("class", "tooltip-icon");
      let toolMood = document.createElement('i');
      toolMood.setAttribute("class", "las la-info-circle");
      let toolSpan = document.createElement('span');
      toolSpan.setAttribute("class", "tooltip-text");
      let featureIndex = featureNames.indexOf(feature);
      let probX, probXAfter, probY, probYAfter, string5;
      // I could make the class names dynamic.
      if (probaDict[featureIndex][probaClickValue + 1] === 0) {
        probX = Math.ceil((1 - probaDict[featureIndex][probaClickValue + 2]) * 100);
        probY = 100 - probX;
        probXAfter = Math.ceil((1 - probaDict[featureIndex][probaClickValue + 3]) * 100);
        probYAfter = 100 - probXAfter;
        let probaDiffCali = Math.ceil(probaDict[featureIndex][probaClickValue + 4] * 100);
        string5 = " It turns out that the chance for "+classNames[1]+" is " + probaDiffCali + "% more!";
      } else {
        probX = Math.ceil(probaDict[featureIndex][probaClickValue + 2] * 100);
        probY = 100 - probX;
        probXAfter = Math.ceil(probaDict[featureIndex][probaClickValue + 3] * 100)
        probYAfter = 100 - probXAfter;
        let probaDiffCali = Math.ceil(probaDict[featureIndex][probaClickValue + 4] * 100);
        string5 = " It turns out that the chance for "+classNames[0]+" is " + probaDiffCali + "% more!";
      }
      let string1 = "Before the change of the value of " + feature + " the chance of "+classNames[0]+" was "  + probX + "%";
      let string2 = " and the chance of "+classNames[1]+" was " + probY + "%.";
      let string3 = " After the change, the chance of "+classNames[0]+" is " + probXAfter + "%";
      let string4 = " and the chance of "+classNames[1]+" is " + probYAfter + "%.";

      toolSpan.innerHTML = string1 + string2 + string3 + string4 + string5;
      toolDiv.appendChild(toolMood);
      toolDiv.appendChild(toolSpan);
      liBot.appendChild(toolDiv);
    }

    botTextFragment.append(liBot);
    document.getElementById('box').appendChild(botTextFragment);
    botTxt.scrollIntoView({
      block: 'start',
      behavior: 'smooth'
    });

    getBotResponse();
  });
}


function changeValueOnClick() {
  feature = featuresDropDown[dropIndex].options[featuresDropDown[dropIndex].selectedIndex].value;

  rawText = 'Choose!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);

  $.when($.ajax(getBotResponse())).then(function() {
    fetchNewFeatureValues();
  });
}


function fetchNewFeatureValues() {
  let optionsFragment = document.createDocumentFragment();
  let optionsLI = document.createElement("li");
  optionsLI.setAttribute("class", "pill-container");

  $.get("/getNewFeatureValues", {
    feature: feature
  }).done(function(data) {
    let categoricalValue = "";
    for (let i = 0; i < data.values.length; i++) {
      if (data.values[i] != 'impossible') {
        let valuePill = document.createElement("button");
        valuePill.setAttribute("class", "pill btn btn-light btn-outline-secondary");
        if (categoricalFeatures.includes(feature)) {
            if (i==1) {
                continue;
            } else {
                data.values[i] = Math.round(data.values[i]);
                categoricalValue = categoricalMap[feature][data.values[i]];
                if (categoricalValue) {
                    valuePill.innerHTML = categoricalValue;
                    valuePill.setAttribute("value", data.values[i].toString());
                    valuePill.addEventListener('click', valuePillClick, false);
                    optionsLI.style.width = previousBotText.offsetWidth + "px";
                    optionsLI.appendChild(valuePill);
                }
            }
        } else {
            valuePill.innerHTML = data.values[i].toString();
            valuePill.setAttribute("value", data.values[i].toString());
            valuePill.addEventListener('click', valuePillClick, false);
            optionsLI.style.width = previousBotText.offsetWidth + "px";
            optionsLI.appendChild(valuePill);
        }

      }
    }
    optionsFragment.appendChild(optionsLI);
    document.getElementById("box").appendChild(optionsFragment);
    optionsLI.scrollIntoView({
      block: 'start',
      behavior: 'smooth'
    });
  });
}

function valuePillClick(e) {
  rawText = 'Change!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  fetchPrediction(1, e.target.value);
  if (flowIndex === 12) {
    $.get("/setDialogFlowIndex", {
      flow_index: 7
    }).done(function(data) {
      flowIndex = parseInt(data);
    });
  }
}

let probaDict = {};


let newFeatDrop = document.createElement("select");
newFeatDrop.setAttribute("id", "newFeatDrop");
newFeatDrop.setAttribute("class", "custom-select");

function fetchChangesInProbabilities() {
  $.get("/getProbabilities").done(function(data) {
    Object.keys(data).forEach(function(key) {
      if (data[key].length != 0) {
        probaDict[key] = [data[key][0], data[key][1], data[key][2], data[key][3], data[key][5]];
      }
    });
    console.log(probaDict);
    Object.keys(probaDict).forEach(function(key) {
      let option = document.createElement("option");
      option.setAttribute("value", featureNames[key]);
      option.innerHTML = featureNames[key];
      newFeatDrop.appendChild(option);
    });

    let inputFragment = document.createDocumentFragment();
    let liWrap = document.createElement("li");
    liWrap.setAttribute("class", "input-container");
    let formWrap = document.createElement("form");
    let inGroup = document.createElement("div");
    inGroup.setAttribute("class", "form-group input-group");
    let prepend = document.createElement("div");
    prepend.setAttribute("class", "input-group-prepend");
    let label = document.createElement("label");
    label.setAttribute("class", "input-group-text");
    label.innerHTML = "Feature:"
    label.setAttribute("for", "newFeatDrop");
    formWrap.appendChild(inGroup);
    inGroup.appendChild(prepend);
    prepend.appendChild(label);
    inGroup.appendChild(newFeatDrop);
    formWrap.appendChild(inGroup);
    let submitBut = document.createElement("button");
    submitBut.setAttribute("type", "button");
    submitBut.setAttribute("class", "btn btn-primary");
    submitBut.innerHTML = "Choose!";
    submitBut.addEventListener('click', probaSubmit, false);
    formWrap.appendChild(submitBut);
    liWrap.appendChild(formWrap)
    inputFragment.appendChild(liWrap);
    document.getElementById("box").appendChild(inputFragment);
    submitBut.scrollIntoView({
      block: 'start',
      behavior: 'smooth'
    });
  });
}

function probaSubmit() {
  feature = newFeatDrop.options[newFeatDrop.selectedIndex].value;
  rawText = 'Choose!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  $.when($.ajax(getBotResponse())).then(function() {
    newSubClick();
  });
}

function newSubClick() {
  let optionsFragment = document.createDocumentFragment();
  let optionsLI = document.createElement("li");
  optionsLI.setAttribute("class", "pill-container");
  let key = featureNames.indexOf(feature);
  for (let i = 0; i < probaDict[key].length * 3; i += probaDict[key].length) {
    if (typeof probaDict[key][i] === 'undefined') {
      break;
    }
    let valuePill = document.createElement("button");
    valuePill.setAttribute("class", "pill btn btn-light btn-outline-secondary");
    let categoricalValue = "";
    if (categoricalFeatures.includes(feature)) {
      console.log(probaDict[key][i]);
      probaDict[key][i] = Math.round(probaDict[key][i]);

      if (probaDict[key][i] >= categoricalMap[feature].length) {
            probaDict[key][i]--;
      }
      categoricalValue = categoricalMap[feature][probaDict[key][i]];
      valuePill.innerHTML = categoricalValue;
      valuePill.setAttribute("value", probaDict[key][i].toString());
      valuePill.addEventListener('click', probaClick, false);
      optionsLI.style.width = previousBotText.offsetWidth + "px";
      optionsLI.appendChild(valuePill);
    }else {
        valuePill.innerHTML = probaDict[key][i].toString();
        valuePill.setAttribute("value", probaDict[key][i].toString());
        valuePill.addEventListener('click', probaClick, false);
        optionsLI.style.width = previousBotText.offsetWidth + "px";
        optionsLI.appendChild(valuePill);
    }


  }
  optionsFragment.appendChild(optionsLI);
  document.getElementById("box").appendChild(optionsFragment);
  optionsLI.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
}

function probaClick(e) {
  rawText = 'Change!'
  let userTextFragment = document.createDocumentFragment();
  let li = document.createElement("li");
  li.setAttribute("class", "user-msg-container");
  let imgDiv = document.createElement("div");
  imgDiv.setAttribute("class", "user-img");
  let img = document.createElement("img");
  img.setAttribute("src", "static/images/hooman_new.png");
  let userTxt = document.createElement("div");
  userTxt.setAttribute("class", "user-msg card");
  userTxt.innerHTML = rawText;
  userTextFragment.appendChild(li);
  li.appendChild(imgDiv);
  imgDiv.appendChild(img);
  li.appendChild(userTxt);
  userTextFragment.append(li);
  document.getElementById('box').appendChild(userTextFragment);
  userTxt.scrollIntoView({
    block: 'start',
    behavior: 'smooth'
  });
  fetchPrediction(2, e.target.value/*, probaDict[featureNames.indexOf(feature)].indexOf(e.target.value)*/);
}