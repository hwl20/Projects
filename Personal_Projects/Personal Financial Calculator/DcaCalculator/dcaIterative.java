package DcaCalculator;

import java.util.Scanner;

public class dcaIterative {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Change in amount? \n\t 1 - no \n\t 2 - yes");
        System.out.print("\nResponse: ");
        int num = scanner.nextInt();
        switch (num){
            case 1:
//                dca case1 = new dca();
//                case1.tryout();
                dca.tryout();
                break;
            case 2:
                System.out.println("Total number of changes (2/3)? \nIf more than 3, indicate number of changes");
                System.out.print("Response: ");
                int changes = scanner.nextInt();
                switch (changes){
                    case 2:
                        dca2 case2 = new dca2();
                        case2.tryout2();
                        break;
                    case 3:
                        dca3 case3 = new dca3();
                        case3.tryout3();
                        break;
                    default:
                        dca4.tryout4(changes);
                }
                break;
        }
    }
}
