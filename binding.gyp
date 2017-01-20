{
	"targets": [
		{
			"target_name": "addon-floatfann",
			"sources": [
				"src/addon-floatfann.cc",
				"src/fanny.cc"
			],
			"libraries": [
				"-lfloatfann"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ]
		}
	]
}
