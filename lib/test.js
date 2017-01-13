var fanny = require('../build/Release/addon');

var FANN = fanny.FANN;

var fann = new FANN();

var res = fann.test([ "foo", "bar", "baz" ], function(err, res) {
	console.log('Async return: ', err, res);
});
console.log('Sync return: ', res);

