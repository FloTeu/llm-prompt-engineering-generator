read -p "Have you already updated the version? If no, patch version update will be performed (yes/no): " response

if [[ "$response" == "yes" ]]; then
    # Your code here
    echo "Build package..."
    hatch build
    echo "Publish package..."
    hatch publish -u __token__
else
    echo "Update version..."
    hatch version patch
    echo "Build package..."
    hatch build
    echo "Publish package..."
    hatch publish -u __token__
fi
