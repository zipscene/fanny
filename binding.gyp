{
	"targets": [
		{
			"target_name": "fanny",
			"sources": [
				"src/fanny.cc"
			],
			"include_dirs": [ "<!(node -e \"require('nan')\")" ]
		}
	]
}
