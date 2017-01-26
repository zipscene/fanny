var addonPath = '../build/Release/';
require(addonPath + 'addon-floatfann');

var XError = require('xerror');

function getAddon(datatype) {
	if (!datatype) datatype = 'float';
	if (datatype === 'float' || datatype === 'double' || datatype === 'fixed') {
		return require(addonPath + 'addon-' + datatype + 'fann');
	} else {
		throw new XError(XError.INVALID_ARGUMENT, 'Invalid FANN datatype: ' + datatype);
	}
}

module.exports = {
	getAddon: getAddon
};

