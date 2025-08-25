// Ensure fonts are loaded properly
document.addEventListener('DOMContentLoaded', function() {
    // Add font loading classes to ensure proper rendering
    document.documentElement.classList.add('fonts-loaded');
    
    // Optional: Use Font Face Observer for better font loading control
    if (window.FontFaceObserver) {
        const fontSansLight = new FontFaceObserver('Source Sans 3', {
            weight: 300
        });
        
        const fontSansSemibold = new FontFaceObserver('Source Sans 3', {
            weight: 600
        });
        
        const fontMono = new FontFaceObserver('Source Code Pro', {
            weight: 400
        });
        
        Promise.all([
            fontSansLight.load(),
            fontSansSemibold.load(),
            fontMono.load()
        ]).then(function() {
            document.documentElement.classList.add('fonts-ready');
        }).catch(function() {
            console.log('Font loading failed, using fallback fonts');
        });
    }
});