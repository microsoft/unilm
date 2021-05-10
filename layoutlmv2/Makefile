.PHONY: quality style
check_dirs := src experiments
# Check that source code meets quality standards

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically

style:
	black $(check_dirs)
	isort $(check_dirs)
