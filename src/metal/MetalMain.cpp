#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include "MetalRenderer.hpp"
#include "Slider.hpp"
#include "Emitter.hpp"
#include "MetalContext.hpp"
#include "MetalGrid.hpp"


int main()
{
    
    //Set Seed
    constexpr int seed = 42;

    //SET WIDTH AND HEIGHT
    constexpr unsigned int width = 512;
    constexpr unsigned int height = 512;    

    constexpr std::size_t cellCount = width*height;
    constexpr std::size_t bytesize = sizeof(float)*cellCount;
    
    //timestep
    constexpr float dt = 1/15.f;

    //clear buffer flag
    bool clear = false;

    //autorelease pool
    NS::AutoreleasePool* pool =
        NS::AutoreleasePool::alloc()->init();


    //mutlithreaded switch
    bool threaded = true;

    //velocity arrow visualization
    bool arrow_viz = false;
    bool paused = false;
    
    //Font loading
    sf::Font font;
    bool loaded = font.openFromFile("./assets/digital-7 (italic).ttf");

    if(!loaded){
    std::cerr << "Failed to load font\n";
    return EXIT_FAILURE;
    }


    Emitter magenta{sf::Vector2f(width*.4f,height*.7f), 50.f, sf::Vector3f(1.f,0.f,.05f)};
    Emitter green{sf::Vector2f(width*.5f,height*.7f), 50.f, sf::Vector3f(0.f,1.f,0.05f)};
    Emitter cyan{sf::Vector2f(width*.6f,height*.7f), 50.f, sf::Vector3f(0.0f,.02f,05.f)};

    std::vector<Emitter*> emitters{
        &magenta,
        &green,
        &cyan
    };

    //OBJECT CREATION
    MetalContext metalcontext{"build/FluidKernels.metallib"};
    MetalGrid metalgrid{width, height, cellCount, bytesize, emitters, seed, metalcontext};
    MetalRenderer renderer(width, height, font);
    // create the window
    sf::RenderWindow window(sf::VideoMode({width, height}), "Wind Sim");
    // create a clock to track the elapsed time
    sf::Clock clock;
    
    
    //SLIDER UI//

    //Grabbing slider controlled variable references
    float* buoyancy = &metalgrid.m_buoyancy;
    float* diffusion = &metalgrid.m_diff_co;
    float* decay = &metalgrid.m_decay;
    float* curl_mult = &metalgrid.m_curl_mult;
    float* noise_freq = &metalgrid.m_noise_freq;
    

    //Slider placement variable
    float slider_right = 130.f;
    
    //Sliders
    Slider buoyancy_slider("Buoyancy", {width-slider_right, 20.f},0.f, 100.f, buoyancy);
    Slider diffusion_slider("Diffusion", {width-slider_right, 50.f},0.f,20.f, diffusion);
    Slider decay_slider("Decay", {width-slider_right, 80.f},0.95f, 0.9999f, decay);
    Slider curl_mult_slider("Noise Strength", {width-slider_right, 110.f},0.f, 2000.f, curl_mult);
    Slider noise_freq_slider("Noise Freq.", {width-slider_right, 140.f},0.025f, .07f, noise_freq);
    
    //Active slider state buffer
   Slider* active_slider = nullptr;

    //Sliders vector container
    std::vector<Slider*> Sliders{
        &buoyancy_slider,
        &diffusion_slider,
        &decay_slider,
        &curl_mult_slider,
        &noise_freq_slider
        
    };

    //CONTROL DISPLAYS

    sf::Text pauseControl(font);
    pauseControl.setCharacterSize(12);
    pauseControl.setFillColor(sf::Color::White);
    pauseControl.setPosition({width-width*.2f, (height-height*0.05f)});
    renderer.setString(pauseControl,"Press 'P' to pause");

    sf::Text resetControl(font);
    resetControl.setCharacterSize(12);
    resetControl.setFillColor(sf::Color::White);
    resetControl.setPosition({width-width*.2f, (height-height*0.08f)});
    renderer.setString(resetControl,"Press 'R' to reset");

    sf::Text arrowsControl(font);
    arrowsControl.setCharacterSize(12);
    arrowsControl.setFillColor(sf::Color::White);
    arrowsControl.setPosition({width-width*.3f, (height-height*0.11f)});
    renderer.setString(arrowsControl,"Press 'V' for Velocity Arrows");

    sf::Text multithreadControl(font);
    multithreadControl.setCharacterSize(12);
    multithreadControl.setFillColor(sf::Color::White);
    multithreadControl.setPosition({width-width*.35f, (height-height*0.14f)});
    renderer.setString(multithreadControl,"Press 'M' to toggle Multithreaded");

    sf::Text stepDisplay(font);
    stepDisplay.setCharacterSize(12);
    stepDisplay.setFillColor(sf::Color::White);
    stepDisplay.setPosition({width-width*.5f, (height-height*0.95f)});
    
    
    std::vector<sf::Text> Instructions{
        pauseControl,
        resetControl,
        arrowsControl,
        multithreadControl
       

    };


    
    // run the main loop
    while (window.isOpen())
    {
        //initialize frame autopool
        auto* framePool = NS::AutoreleasePool::alloc()->init();
        
        
        // handle events
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                if (key->code == sf::Keyboard::Key::V) {
                    arrow_viz = !arrow_viz;
                    }
                }

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                if (key->code == sf::Keyboard::Key::P) {
                    paused = !paused;
                    }
                }

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                if (key->code == sf::Keyboard::Key::R) {
                    clear=true;
                    }
                }


            if(event->getIf<sf::Event::MouseButtonPressed>()){
                sf::Vector2i mousePixel = sf::Mouse::getPosition(window);
                sf::Vector2f mousePos = window.mapPixelToCoords(mousePixel);

                for(auto& slider : Sliders){
                    if(slider->contains(mousePos)){
                        active_slider = slider;
                        break;
                    }
                
                }
            }

            if (event->getIf<sf::Event::MouseButtonReleased>()) {
                if (active_slider) {
                    active_slider->dragging = false;
                    active_slider = nullptr;
                }
            }
            
        }
        
        if(active_slider){
            active_slider->dragging = true;
            active_slider->updateFromMouse(window.mapPixelToCoords(sf::Mouse::getPosition(window)));
        }
        

        // UPDATE //
        if(!paused){
            metalgrid.update(dt);
            if(clear){
                metalgrid.ClearBuffers();
                clear=false;
            }
            renderer.update(metalgrid, arrow_viz);
            
        }

        
        // DRAW BLOCK
        
        // clear window
        window.clear();

        renderer.draw(window, arrow_viz);
        
        //slider drawing
        for( auto& slider : Sliders){
            slider->draw(window);
        }

        //draw instructions
        for( auto& instruction : Instructions){
            window.draw(instruction);
        }
        window.draw(stepDisplay);

        // display
        window.display();

        //release frame autopool
        framePool->release();
    }


return 0;
}