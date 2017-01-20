const readline = require("readline"),
	rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout
	});
var globalVar = {
		gridDimension: null,
		belief: []
	},
	methods = {
		initializeWorld: null
	},
	initModule;

/*methods.initializeWorld = function (gridDimension) {

};
methods.setGlobals = function (reference, value) {
	reference = value;
};*/
methods.extractUserInput = function (currentQAPair) {
	rl.question(currentQAPair.question, (answer) => {
		globalVar[currentQAPair.answer] = answer;
		rl.pause();
	});
};
/*methods.setInputValues = function (answers) {
};*/
initModule = function () {
	var initialUnknown = [{
			question: "Enter grid dimensions in m x n format:\t",
			answer: "gridDimension"
			},
			{
			question: "Enter belief:\t",
			answer: "belief"
			}];
	while (initialUnknown.length !== 0) {
		methods.extractUserInput(initialUnknown[0]);
		initialUnknown.splice(0, 1);
	}
	// methods.setInputValues({
	// 	gridDimension: 
	// });
};
initModule();